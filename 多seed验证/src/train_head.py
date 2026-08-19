"""
Head 重训（A/B/C 组）与正常继续训练（D 组）
"""
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data import WaterbirdsDataset, get_transforms


def create_distribution_loader(root_dir, mode, batch_size=32, seed=42, max_samples=None):
    """构造指定分布的训练 loader。

    mode:
      'original'  : 95% aligned / 5% conflicting (A/D 组)
      'balanced'  : 25/25/25/25 均衡 (B 组)
      'inverse'   : 反相关，只用 conflicting (C 组)

    max_samples: 冒烟测试时限制样本数。
    """
    random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = WaterbirdsDataset(root_dir, split='train', transform=get_transforms('train'))

    # 分组
    aligned_ww = []  # 水鸟+水 (aligned)
    aligned_ll = []  # 陆鸟+陆 (aligned)
    conflict_wl = []  # 水鸟+陆 (conflicting)
    conflict_lw = []  # 陆鸟+水 (conflicting)

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

    if mode == 'original':
        # 95% aligned, 5% conflicting
        total = len(train_dataset.samples)
        n_aligned = int(total * 0.95)
        n_conflict = total - n_aligned

        half = n_aligned // 2
        random.shuffle(aligned_ww)
        random.shuffle(aligned_ll)
        chosen_aligned = aligned_ww[:half] + aligned_ll[:half]

        random.shuffle(conflict_wl)
        random.shuffle(conflict_lw)
        conflict_pool = conflict_wl + conflict_lw
        random.shuffle(conflict_pool)
        chosen_conflict = conflict_pool[:n_conflict]

        indices = chosen_aligned + chosen_conflict

    elif mode == 'balanced':
        # 4 组各取 min 长度，均衡采样
        min_len = min(len(aligned_ww), len(aligned_ll), len(conflict_wl), len(conflict_lw))
        random.shuffle(aligned_ww)
        random.shuffle(aligned_ll)
        random.shuffle(conflict_wl)
        random.shuffle(conflict_lw)
        indices = aligned_ww[:min_len] + aligned_ll[:min_len] + conflict_wl[:min_len] + conflict_lw[:min_len]

    elif mode == 'inverse':
        # 反相关，只用 conflicting
        pool = conflict_wl + conflict_lw
        random.shuffle(pool)
        indices = pool
    else:
        raise ValueError(f"未知 mode: {mode}")

    if max_samples is not None:
        indices = indices[:max_samples]

    subset = Subset(train_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


def train_head(model, loader, epochs, lr=0.001, device='cpu', freeze_backbone=False):
    """重训模型（head），可选冻结 backbone。返回训练后的模型。"""
    model.to(device)
    model.train()

    # 冻结 backbone
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  Head重训 epoch {epoch+1}/{epochs} loss={total_loss/max(n_batch,1):.4f}")

    return model
