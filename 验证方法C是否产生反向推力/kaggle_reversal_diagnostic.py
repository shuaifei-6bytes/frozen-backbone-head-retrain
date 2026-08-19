# -*- coding: utf-8 -*-
"""
C-Reversal 诊断实验 — Kaggle 单文件版本
=======================================
在 Kaggle Notebook 里直接运行，所有依赖内联。

前置条件：
1. 添加 Dataset "waterbird-complete95" (feishuai/waterbird-complete95)
   → 挂载到 /kaggle/input/waterbird-complete95/waterbird_complete95_forest2water2/
   或者用你自己的 Kaggle dataset 路径。
2. GPU: 设置 → Accelerator → GPU T4 x2
3. 运行后所有输出在 /kaggle/working/output/，可直接下载。

输出：
  - m_global_seed{X}.pt   模型权重
  - B_seed{X}.pt
  - C_seed{X}.pt
  - results.json           完整诊断结果
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# ============================================================================
# 配置（直接改这里，不需要 config.json）
# ============================================================================
CONFIG = {
    "seeds": [42, 123, 456, 789],
    # Kaggle 数据集路径
    "data_dir": "/kaggle/input/datasets/feishuai/waterbird-complete95/waterbird_complete95_forest2water2",
    "num_clients": 5,
    "federated_rounds": 30,
    "local_epochs": 1,
    "head_epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "num_workers": 4,
    "output_dir": "/kaggle/working/output",
}

# ============================================================================
# 数据集
# ============================================================================
class WaterbirdsDataset(Dataset):
    """Waterbirds 数据集加载器
    metadata.csv 列: img_id, img_filename, y(0水鸟/1陆鸟), split(0train/1val/2test), place(0水/1陆), data_id
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        split_map = {'train': 0, 'val': 1, 'test': 2}
        target_split = split_map[split]
        with open(os.path.join(root_dir, 'metadata.csv'), 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                img_filename = parts[1]
                y = int(parts[2])
                img_split = int(parts[3])
                place = int(parts[4])
                if img_split == target_split:
                    self.samples.append({'img_filename': img_filename, 'y': y, 'place': place})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.root_dir, sample['img_filename'])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, sample['y'], sample['place']


def get_transforms(split='train'):
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# 模型
# ============================================================================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(num_classes=2, pretrained=True, device='cpu'):
    """创建 ResNet-18，backbone + 分类 head"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def copy_model_state(model):
    """深拷贝模型参数到 CPU"""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state_dict):
    """加载状态字典，自动处理设备"""
    device = next(model.parameters()).device
    state = {k: v.to(device) for k, v in state_dict.items()}
    model.load_state_dict(state)


def save_model(model, path):
    """保存模型权重（始终存 CPU 版本）"""
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)


# ============================================================================
# 数据加载器
# ============================================================================
def create_federated_loaders(root_dir, num_clients=5, batch_size=32, seed=42, num_workers=0):
    """为每个客户端创建数据加载器"""
    random.seed(seed)
    torch.manual_seed(seed)
    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    client_loaders = []
    client_size = len(indices) // num_clients
    for i in range(num_clients):
        if i == num_clients - 1:
            ci = indices[i * client_size:]
        else:
            ci = indices[i * client_size:(i + 1) * client_size]
        subset = Subset(train_dataset, ci)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_loaders.append(loader)
    return client_loaders


def get_balanced_test_loader(root_dir, batch_size=64, num_workers=0):
    """创建四组均衡测试加载器（反事实配对）"""
    test_dataset = WaterbirdsDataset(root_dir, split='test', transform=get_transforms('val'))
    groups = {'waterbird_water': [], 'waterbird_land': [], 'landbird_land': [], 'landbird_water': []}
    for idx, sample in enumerate(test_dataset.samples):
        y, place = sample['y'], sample['place']
        if y == 0 and place == 0:
            groups['waterbird_water'].append(idx)
        elif y == 0 and place == 1:
            groups['waterbird_land'].append(idx)
        elif y == 1 and place == 1:
            groups['landbird_land'].append(idx)
        elif y == 1 and place == 0:
            groups['landbird_water'].append(idx)
    group_loaders = {}
    for name, indices in groups.items():
        subset = Subset(test_dataset, indices)
        group_loaders[name] = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return group_loaders


def create_distribution_loader(root_dir, mode, batch_size=32, seed=42, num_workers=0):
    """构造指定分布的训练 loader"""
    random.seed(seed)
    torch.manual_seed(seed)
    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))

    aligned_ww, aligned_ll, conflict_wl, conflict_lw = [], [], [], []
    for idx, sample in enumerate(train_dataset.samples):
        y, p = sample['y'], sample['place']
        if y == 0 and p == 0:
            aligned_ww.append(idx)
        elif y == 1 and p == 1:
            aligned_ll.append(idx)
        elif y == 0 and p == 1:
            conflict_wl.append(idx)
        elif y == 1 and p == 0:
            conflict_lw.append(idx)

    if mode == 'balanced':
        min_len = min(len(aligned_ww), len(aligned_ll), len(conflict_wl), len(conflict_lw))
        for lst in [aligned_ww, aligned_ll, conflict_wl, conflict_lw]:
            random.shuffle(lst)
        indices = aligned_ww[:min_len] + aligned_ll[:min_len] + conflict_wl[:min_len] + conflict_lw[:min_len]
    elif mode == 'inverse':
        pool = conflict_wl + conflict_lw
        random.shuffle(pool)
        indices = pool
    else:
        raise ValueError(f"未知 mode: {mode}")

    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# ============================================================================
# 联邦训练
# ============================================================================
def federated_train(global_model, client_loaders, num_rounds, local_epochs, lr=0.001, device='cpu'):
    """FedAvg 联邦训练循环"""
    for r in range(num_rounds):
        client_states = []
        for loader in client_loaders:
            model = create_model(num_classes=2, pretrained=False, device=device)
            load_state(model, global_model.state_dict())
            model.train()
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for _ in range(local_epochs):
                for images, labels, _ in loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
            client_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        # FedAvg 聚合
        avg_state = {}
        for key in client_states[0]:
            avg_state[key] = sum(sd[key] for sd in client_states) / len(client_states)
        load_state(global_model, avg_state)
        print(f"  联邦轮次 {r+1}/{num_rounds} 完成")


# ============================================================================
# Head 重训
# ============================================================================
def train_head(model, loader, epochs, lr=0.001, device='cpu'):
    """冻结 backbone，只训练 head"""
    model.to(device)
    model.train()
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss, n_batch = 0.0, 0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  Head重训 epoch {epoch+1}/{epochs} loss={total_loss/max(n_batch,1):.4f}")


# ============================================================================
# 方向性诊断
# ============================================================================
@torch.no_grad()
def collect_probs_preds(model, loader, device):
    """收集 softmax 概率和预测"""
    model.to(device)
    model.eval()
    probs_list, preds_list, labels_list = [], [], []
    for images, labels, _ in loader:
        images = images.to(device)
        probs = F.softmax(model(images), dim=1)
        _, preds = torch.max(probs, 1)
        probs_list.append(probs.cpu())
        preds_list.append(preds.cpu())
        labels_list.append(labels)
    if not probs_list:
        return torch.empty(0, 2), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(probs_list, 0), torch.cat(preds_list, 0), torch.cat(labels_list, 0)


def diagnose_model(model, group_loaders, device):
    """对单个模型做完整方向性诊断"""

    # ---- 有符号背景效应 ----
    probs_ww, _, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)
    probs_ll, _, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)

    delta_waterbird = 0.0
    if len(probs_ww) > 0 and len(probs_wl) > 0:
        delta_waterbird = probs_ww[:, 0].mean().item() - probs_wl[:, 0].mean().item()

    delta_landbird = 0.0
    if len(probs_ll) > 0 and len(probs_lw) > 0:
        delta_landbird = probs_ll[:, 1].mean().item() - probs_lw[:, 1].mean().item()

    # ---- 方向性翻转率 ----
    _, preds_ww, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    _, preds_wl, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)
    _, preds_ll, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    _, preds_lw, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)

    original_flips, reverse_flips, total_pairs = 0, 0, 0

    # 水鸟配对
    min_w = min(len(preds_ww), len(preds_wl))
    for i in range(min_w):
        pw, pl = preds_ww[i].item(), preds_wl[i].item()
        total_pairs += 1
        if pw != pl:
            if pw == 0 and pl == 1:
                original_flips += 1
            elif pw == 1 and pl == 0:
                reverse_flips += 1

    # 陆鸟配对
    min_l = min(len(preds_ll), len(preds_lw))
    for i in range(min_l):
        pl, pw = preds_ll[i].item(), preds_lw[i].item()
        total_pairs += 1
        if pl != pw:
            if pl == 1 and pw == 0:
                original_flips += 1
            elif pl == 0 and pw == 1:
                reverse_flips += 1

    orig_flip_rate = original_flips / total_pairs if total_pairs > 0 else 0.0
    rev_flip_rate = reverse_flips / total_pairs if total_pairs > 0 else 0.0

    # ---- 四种 Group Accuracy ----
    group_accs = {}
    for name, loader in group_loaders.items():
        correct, total = 0, 0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        group_accs[name] = correct / total if total > 0 else 0.0

    # ---- 每样本 signed effect ----
    wb_effects = []
    for i in range(min(len(probs_ww), len(probs_wl))):
        wb_effects.append(float(probs_ww[i, 0] - probs_wl[i, 0]))
    lb_effects = []
    for i in range(min(len(probs_ll), len(probs_lw))):
        lb_effects.append(float(probs_ll[i, 1] - probs_lw[i, 1]))

    return {
        'delta_waterbird': float(delta_waterbird),
        'delta_landbird': float(delta_landbird),
        'original_flip_rate': float(orig_flip_rate),
        'reverse_flip_rate': float(rev_flip_rate),
        'group_accuracy': {k: float(v) for k, v in group_accs.items()},
        'per_sample': {
            'waterbird_effects': wb_effects,
            'landbird_effects': lb_effects,
        }
    }


# ============================================================================
# 主流程
# ============================================================================
def run_single_seed(config, seed, device, output_dir):
    """运行单个 seed 的完整实验"""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = config['data_dir']
    batch_size = config['batch_size']
    lr = config['lr']
    num_workers = config['num_workers']

    # 阶段1: FedAvg 训练 M_global
    print("\n[阶段1] FedAvg 联邦训练 M_global...")
    m_global = create_model(num_classes=2, pretrained=True, device=device)
    client_loaders = create_federated_loaders(
        data_dir, num_clients=config['num_clients'], batch_size=batch_size,
        seed=seed, num_workers=num_workers
    )
    federated_train(m_global, client_loaders, config['federated_rounds'],
                    config['local_epochs'], lr=lr, device=device)

    m_global_path = os.path.join(output_dir, f'm_global_seed{seed}.pt')
    save_model(m_global, m_global_path)
    print(f"  M_global 权重已保存: {m_global_path}")

    m_global_state = copy_model_state(m_global)

    # 阶段2: B 组（均衡背景 Head 重训）
    print("\n[阶段2] B 组：均衡背景 Head 重训 (冻结 backbone)")
    model_b = create_model(num_classes=2, pretrained=False, device=device)
    load_state(model_b, m_global_state)
    loader_b = create_distribution_loader(data_dir, 'balanced', batch_size, seed, num_workers)
    train_head(model_b, loader_b, config['head_epochs'], lr=lr, device=device)

    model_b_path = os.path.join(output_dir, f'B_seed{seed}.pt')
    save_model(model_b, model_b_path)
    print(f"  B 组权重已保存: {model_b_path}")

    # 阶段3: C 组（反相关 Head 重训）
    print("\n[阶段3] C 组：反相关 Head 重训 (冻结 backbone)")
    model_c = create_model(num_classes=2, pretrained=False, device=device)
    load_state(model_c, m_global_state)
    loader_c = create_distribution_loader(data_dir, 'inverse', batch_size, seed, num_workers)
    train_head(model_c, loader_c, config['head_epochs'], lr=lr, device=device)

    model_c_path = os.path.join(output_dir, f'C_seed{seed}.pt')
    save_model(model_c, model_c_path)
    print(f"  C 组权重已保存: {model_c_path}")

    # 阶段4: 方向性诊断
    print("\n[阶段4] 方向性诊断")
    group_loaders = get_balanced_test_loader(data_dir, batch_size=64, num_workers=num_workers)

    diagnostics = {}
    for tag, model in [('M_global', m_global), ('B', model_b), ('C', model_c)]:
        print(f"  诊断 {tag}...")
        diag = diagnose_model(model, group_loaders, device)
        diagnostics[tag] = diag
        print(f"    Δ_waterbird = {diag['delta_waterbird']:+.4f}")
        print(f"    Δ_landbird  = {diag['delta_landbird']:+.4f}")
        print(f"    Original Flip = {diag['original_flip_rate']*100:.2f}%")
        print(f"    Reverse Flip  = {diag['reverse_flip_rate']*100:.2f}%")
        ga = diag['group_accuracy']
        print(f"    WB-Water={ga.get('waterbird_water',0)*100:.2f}% "
              f"WB-Land={ga.get('waterbird_land',0)*100:.2f}% "
              f"LB-Land={ga.get('landbird_land',0)*100:.2f}% "
              f"LB-Water={ga.get('landbird_water',0)*100:.2f}%")

    # 阶段5: 打印该 seed 表格
    print(f"\n{'='*60}")
    print(f"Seed {seed} 诊断结果")
    print(f"{'='*60}")
    print(f"{'模型':<12}{'Δ_waterbird':>14}{'Δ_landbird':>14}{'Orig Flip%':>14}{'Rev Flip%':>14}")
    print("-" * 68)
    for tag in ['M_global', 'B', 'C']:
        d = diagnostics[tag]
        print(f"{tag:<12}{d['delta_waterbird']:>+14.4f}{d['delta_landbird']:>+14.4f}"
              f"{d['original_flip_rate']*100:>13.2f}%{d['reverse_flip_rate']*100:>13.2f}%")

    print(f"\n{'模型':<12}{'WB-Water':>12}{'WB-Land':>12}{'LB-Land':>12}{'LB-Water':>12}")
    print("-" * 60)
    for tag in ['M_global', 'B', 'C']:
        ga = diagnostics[tag]['group_accuracy']
        print(f"{tag:<12}{ga.get('waterbird_water',0)*100:>11.2f}%"
              f"{ga.get('waterbird_land',0)*100:>11.2f}%"
              f"{ga.get('landbird_land',0)*100:>11.2f}%"
              f"{ga.get('landbird_water',0)*100:>11.2f}%")

    return {'seed': seed, 'diagnostics': diagnostics}


def main():
    device = get_device()
    print(f"使用设备: {device}")
    print(f"种子列表: {CONFIG['seeds']}")
    print(f"数据目录: {CONFIG['data_dir']}")

    # 创建输出目录
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 运行多 seed 实验
    all_results = []
    results_path = os.path.join(output_dir, 'results.json')

    for seed in CONFIG['seeds']:
        try:
            result = run_single_seed(CONFIG, seed, device, output_dir)
            all_results.append(result)
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Seed {seed} 完成，结果已保存")
        except Exception as e:
            print(f"\n✗ Seed {seed} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'seed': seed, 'error': str(e)})
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 多 seed 汇总
    valid = [r for r in all_results if 'error' not in r]
    if valid:
        print(f"\n{'='*60}")
        print("多 Seed 汇总 (Mean ± Std)")
        print(f"{'='*60}")

        tags = ['M_global', 'B', 'C']
        metrics = ['delta_waterbird', 'delta_landbird', 'original_flip_rate', 'reverse_flip_rate']
        group_keys = ['waterbird_water', 'waterbird_land', 'landbird_land', 'landbird_water']

        summary = {}
        for tag in tags:
            summary[tag] = {}
            for metric in metrics:
                vals = [r['diagnostics'][tag][metric] for r in valid]
                summary[tag][metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
            for gk in group_keys:
                vals = [r['diagnostics'][tag]['group_accuracy'].get(gk, 0) for r in valid]
                summary[tag][f'group_acc_{gk}'] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

        print(f"\n{'模型':<12}{'Δ_waterbird':>20}{'Δ_landbird':>20}{'Orig Flip%':>20}{'Rev Flip%':>20}")
        print("-" * 92)
        for tag in tags:
            s = summary[tag]
            print(f"{tag:<12}"
                  f"{s['delta_waterbird']['mean']:+.4f}±{s['delta_waterbird']['std']:.4f}"
                  f"{'':>4}{s['delta_landbird']['mean']:+.4f}±{s['delta_landbird']['std']:.4f}"
                  f"{'':>4}{s['original_flip_rate']['mean']*100:.2f}±{s['original_flip_rate']['std']*100:.2f}%"
                  f"{'':>4}{s['reverse_flip_rate']['mean']*100:.2f}±{s['reverse_flip_rate']['std']*100:.2f}%")

        print(f"\n{'模型':<12}{'WB-Water':>20}{'WB-Land':>20}{'LB-Land':>20}{'LB-Water':>20}")
        print("-" * 92)
        for tag in tags:
            s = summary[tag]
            print(f"{tag:<12}"
                  f"{s['group_acc_waterbird_water']['mean']*100:.2f}±{s['group_acc_waterbird_water']['std']*100:.2f}%"
                  f"{'':>4}{s['group_acc_waterbird_land']['mean']*100:.2f}±{s['group_acc_waterbird_land']['std']*100:.2f}%"
                  f"{'':>4}{s['group_acc_landbird_land']['mean']*100:.2f}±{s['group_acc_landbird_land']['std']*100:.2f}%"
                  f"{'':>4}{s['group_acc_landbird_water']['mean']*100:.2f}±{s['group_acc_landbird_water']['std']*100:.2f}%")

        # 保存含汇总的最终结果
        final = {'per_seed': all_results, 'summary': summary}
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("实验完成！")
    print(f"  结果文件: {results_path}")
    print(f"  权重目录: {output_dir}/")
    print(f"  在 Kaggle 右侧 Output 面板可直接下载")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
