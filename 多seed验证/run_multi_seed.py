"""
主实验脚本：多seed验证 - 最小批次共识验证门控稀疏关系编辑
实验流程：
1. 从 M_global 开始
2. 提取 conflicting samples，按多个 minibatch 切分
3. 对每个 minibatch 计算 Backbone 通道的 relation attribution
4. 跨 minibatch 共识聚合，得到通道重要性排名
5. 选择 Top-K 通道，冻结其余参数，只编辑 Top-K 通道
6. 在 edit data 上训练候选模型
7. 验证门控：删除提升 ≥ tau_D 且保留损失 ≤ tau_P
8. 对照实验：随机 Top-K、Head-only
9. 多 seed 循环，记录结果
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import create_model, get_device
from src.data import WaterbirdsDataset, create_federated_loaders, get_transforms
from src.federated import federated_train
from src.head_only import train_head_only
from src.evaluate import evaluate_model, compute_background_gap_real
from src.attribution import compute_minibatch_attribution, compute_all_minibatches_attribution
from src.consensus import aggregate_consensus
from src.sparse_edit import apply_sparse_edit
from src.validation_gate import validate_candidate


def load_config(config_path):
    """加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def prepare_data(config, seed, device):
    """准备数据加载器"""
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    
    # 设置数据加载的随机种子
    torch.manual_seed(seed)
    
    # 创建数据集
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = WaterbirdsDataset(data_dir, split='train', transform=train_transform)
    test_dataset = WaterbirdsDataset(data_dir, split='test', transform=val_transform)
    
    # 创建 conflicting samples 加载器（用于归因）
    conflicting_samples = [
        idx for idx, sample in enumerate(train_dataset.samples)
        if (sample['y'] == 0 and sample['place'] == 1) or
           (sample['y'] == 1 and sample['place'] == 0)
    ]
    
    conflicting_dataset = torch.utils.data.Subset(train_dataset, conflicting_samples)
    conflicting_loader = torch.utils.data.DataLoader(
        conflicting_dataset,
        batch_size=config['mb_batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    # 创建 edit data 加载器（均衡分布，用于编辑）
    balanced_indices = []
    for group in ['waterbird_water', 'waterbird_land', 'landbird_land', 'landbird_water']:
        indices = [
            idx for idx, sample in enumerate(train_dataset.samples)
            if (sample['y'] == 0 and sample['place'] == 0 and group == 'waterbird_water') or
               (sample['y'] == 0 and sample['place'] == 1 and group == 'waterbird_land') or
               (sample['y'] == 1 and sample['place'] == 1 and group == 'landbird_land') or
               (sample['y'] == 1 and sample['place'] == 0 and group == 'landbird_water')
        ]
        # 每组取 min_size 个样本
        min_size = min(len(indices), 200)
        balanced_indices.extend(indices[:min_size])
    
    balanced_dataset = torch.utils.data.Subset(train_dataset, balanced_indices)
    balanced_loader = torch.utils.data.DataLoader(
        balanced_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 0)
    )
    
    # 创建测试加载器
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 0)
    )
    
    return conflicting_loader, balanced_loader, test_loader


def run_single_seed(config, seed, device):
    """运行单个 seed 的实验"""
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 准备数据
    conflicting_loader, balanced_loader, test_loader = prepare_data(config, seed, device)
    
    # 加载 M_global（从之前的实验加载，或重新训练）
    print("\n[阶段1] 加载 M_global...")
    m_global = create_model(num_classes=2, pretrained=False, device=device)
    
    # 如果有保存的 M_global，加载它
    m_global_path = config.get('m_global_path')
    if m_global_path and os.path.exists(m_global_path):
        state_dict = torch.load(m_global_path, map_location=device)
        m_global.load_state_dict(state_dict)
        print(f"从 {m_global_path} 加载 M_global")
    else:
        print("警告：未找到 M_global，使用随机初始化的模型")
    
    # 评估 M_global 基线指标
    print("\n[阶段2] 评估 M_global 基线指标...")
    oa_orig, wg_orig, _ = evaluate_model(m_global, test_loader, device)
    bg_orig, flip_orig = compute_background_gap_real(m_global, test_loader, device)
    print(f"M_global - OA: {oa_orig:.4f}, WG: {wg_orig:.4f}, BG: {bg_orig:.4f}, Flip Rate: {flip_orig:.4f}")
    
    # 阶段3：多 minibatch 归因
    print(f"\n[阶段3] 多 minibatch 归因（{config['num_minibatches']} 个 minibatch）...")
    mb_scores = compute_all_minibatches_attribution(
        m_global,
        conflicting_loader,
        num_minibatches=config['num_minibatches'],
        device=device
    )
    
    # 阶段4：跨 minibatch 共识聚合
    print("\n[阶段4] 跨 minibatch 共识聚合...")
    consensus = aggregate_consensus(mb_scores)
    print(f"共识聚合完成，共 {len(consensus)} 个通道")
    
    # 阶段5：Top-K 选择
    top_k = config['top_k']
    top_k_channels = consensus[:top_k]
    print(f"\n[阶段5] Top-{top_k} 选择")
    for i, (layer, channel, score) in enumerate(top_k_channels[:5]):
        print(f"  {i+1}. {layer} channel {channel}: {score:.4f}")
    if len(top_k_channels) > 5:
        print(f"  ... 共 {len(top_k_channels)} 个通道")
    
    # 阶段6：稀疏编辑
    print(f"\n[阶段6] 稀疏编辑（Top-{top_k} 通道）...")
    m_candidate = apply_sparse_edit(
        m_global,
        top_k_channels,
        balanced_loader,
        epochs=config['sparse_edit_epochs'],
        lr=config['sparse_edit_lr'],
        device=device
    )
    
    # 阶段7：验证门控
    print("\n[阶段7] 验证门控...")
    oa_cand, wg_cand, _ = evaluate_model(m_candidate, test_loader, device)
    bg_cand, flip_cand = compute_background_gap_real(m_candidate, test_loader, device)
    bg_drop = bg_orig - bg_cand
    oa_drop = oa_orig - oa_cand
    flip_drop = flip_orig - flip_cand
    
    deletion_pass = bg_drop >= config['tau_D']
    preservation_pass = oa_drop <= config['tau_P']
    passed = deletion_pass and preservation_pass
    
    print(f"  Background Gap: {bg_orig:.4f} → {bg_cand:.4f} (drop: {bg_drop:.4f}, threshold: {config['tau_D']})")
    print(f"  Flip Rate: {flip_orig:.4f} → {flip_cand:.4f} (drop: {flip_drop:.4f})")
    print(f"  Overall Accuracy: {oa_orig:.4f} → {oa_cand:.4f} (drop: {oa_drop:.4f}, threshold: {config['tau_P']})")
    print(f"  删除通过: {deletion_pass}, 保留通过: {preservation_pass}, 总体通过: {passed}")
    
    # 阶段8：对照实验
    print("\n[阶段8] 对照实验...")
    
    # 8.1 随机 Top-K
    print("  8.1 随机 Top-K 对照...")
    import random
    random.seed(seed)
    all_channels = [(layer, channel) for layer, module in m_global.named_modules() 
                    if hasattr(module, 'out_channels') 
                    for channel in range(module.out_channels)]
    random_channels = random.sample(all_channels, min(top_k, len(all_channels)))
    
    m_random = apply_sparse_edit(
        m_global,
        random_channels,
        balanced_loader,
        epochs=config['sparse_edit_epochs'],
        lr=config['sparse_edit_lr'],
        device=device
    )
    oa_random, wg_random, _ = evaluate_model(m_random, test_loader, device)
    bg_random, flip_random = compute_background_gap_real(m_random, test_loader, device)
    bg_drop_random = bg_orig - bg_random
    oa_drop_random = oa_orig - oa_random
    random_passed = (bg_drop_random >= config['tau_D']) and (oa_drop_random <= config['tau_P'])
    print(f"    随机 Top-K - OA: {oa_random:.4f}, BG: {bg_random:.4f}, Flip Rate: {flip_random:.4f}, passed: {random_passed}")
    
    # 8.2 Head-only
    print("  8.2 Head-only 对照...")
    m_headonly = train_head_only(
        m_global,
        balanced_loader,
        epochs=config['head_only_epochs'],
        lr=config['head_only_lr'],
        device=device
    )
    oa_headonly, wg_headonly, _ = evaluate_model(m_headonly, test_loader, device)
    bg_headonly, flip_headonly = compute_background_gap_real(m_headonly, test_loader, device)
    bg_drop_headonly = bg_orig - bg_headonly
    oa_drop_headonly = oa_orig - oa_headonly
    headonly_passed = (bg_drop_headonly >= config['tau_D']) and (oa_drop_headonly <= config['tau_P'])
    print(f"    Head-only - OA: {oa_headonly:.4f}, BG: {bg_headonly:.4f}, Flip Rate: {flip_headonly:.4f}, passed: {headonly_passed}")
    
    # 返回结果
    result = {
        'seed': seed,
        'm_global': {
            'overall_acc': float(oa_orig),
            'worst_group_acc': float(wg_orig),
            'background_gap': float(bg_orig),
            'flip_rate': float(flip_orig),
        },
        'sparse_edit_top_k': {
            'overall_acc': float(oa_cand),
            'worst_group_acc': float(wg_cand),
            'background_gap': float(bg_cand),
            'flip_rate': float(flip_cand),
            'bg_drop': float(bg_drop),
            'oa_drop': float(oa_drop),
            'flip_drop': float(flip_drop),
            'passed': bool(passed),
            'top_k_channels': [(layer, int(ch)) for layer, ch, _ in top_k_channels],
        },
        'random_top_k': {
            'overall_acc': float(oa_random),
            'worst_group_acc': float(wg_random),
            'background_gap': float(bg_random),
            'flip_rate': float(flip_random),
            'bg_drop': float(bg_drop_random),
            'oa_drop': float(oa_drop_random),
            'passed': bool(random_passed),
        },
        'head_only': {
            'overall_acc': float(oa_headonly),
            'worst_group_acc': float(wg_headonly),
            'background_gap': float(bg_headonly),
            'flip_rate': float(flip_headonly),
            'bg_drop': float(bg_drop_headonly),
            'oa_drop': float(oa_drop_headonly),
            'passed': bool(headonly_passed),
        },
    }
    
    return result


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多seed验证实验')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出 JSON 路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    device = get_device(config.get('device', 'auto'))
    
    print(f"使用设备: {device}")
    print(f"种子列表: {config['seeds']}")
    print(f"Top-K: {config['top_k']}")
    print(f"Minibatch 数量: {config['num_minibatches']}")
    
    # 运行多 seed 实验
    all_results = []
    
    for seed in config['seeds']:
        try:
            result = run_single_seed(config, seed, device)
            all_results.append(result)
            
            # 每完成一个 seed 就保存一次
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Seed {seed} 完成，结果已保存到 {args.output}")
            
        except Exception as e:
            print(f"\n✗ Seed {seed} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'seed': seed,
                'error': str(e),
            })
    
    # 最终保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("实验汇总")
    print(f"{'='*60}")
    
    passed_count = {
        'sparse_edit_top_k': sum(1 for r in all_results if r.get('sparse_edit_top_k', {}).get('passed', False)),
        'random_top_k': sum(1 for r in all_results if r.get('random_top_k', {}).get('passed', False)),
        'head_only': sum(1 for r in all_results if r.get('head_only', {}).get('passed', False)),
    }
    
    total_seeds = len([s for s in config['seeds'] if any(r['seed'] == s and 'error' not in r for r in all_results)])
    
    print(f"成功运行的 seed 数: {total_seeds}")
    print(f"Sparse Edit Top-K 通过数: {passed_count['sparse_edit_top_k']}/{total_seeds}")
    print(f"Random Top-K 通过数: {passed_count['random_top_k']}/{total_seeds}")
    print(f"Head-only 通过数: {passed_count['head_only']}/{total_seeds}")
    
    if total_seeds > 0:
        print(f"\n通过率:")
        print(f"  Sparse Edit Top-K: {passed_count['sparse_edit_top_k']/total_seeds*100:.1f}%")
        print(f"  Random Top-K: {passed_count['random_top_k']/total_seeds*100:.1f}%")
        print(f"  Head-only: {passed_count['head_only']/total_seeds*100:.1f}%")
    
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
