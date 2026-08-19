"""
C-Reversal 诊断实验主脚本
========================
实验目的：判断反相关 Head 重训（C 组）是真的去除了背景依赖，还是把原来的
         "背景→类别"正向关系推成了反向关系。

实验流程（每个 seed）：
  阶段1: FedAvg 联邦训练 M_global，保存权重
  阶段2: B 组（均衡背景 Head 重训，冻结 backbone），保存权重
  阶段3: C 组（反相关 Head 重训，冻结 backbone），保存权重
  阶段4: 加载 M_global/B/C 权重，用相同反事实样本做方向性诊断
  阶段5: 输出该 seed 的指标表格

多 seed 结束后计算 Mean±Std 汇总。

用法:
  python run_reversal_diagnostic.py --config config.json --output results.json
  python run_reversal_diagnostic.py --config config.json --smoke   # 冒烟测试

Kaggle 用法:
  python run_reversal_diagnostic.py --config config.json --output results.json --device auto --num_workers 4
"""
import argparse
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# 添加 src 到 path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.model import create_model, get_device, copy_model_state, load_state, save_model, load_model
from src.data import (
    WaterbirdsDataset, create_federated_loaders, get_transforms,
    get_test_loader, get_balanced_test_loader, create_distribution_loader
)
from src.federated import federated_train
from src.train_head import train_head
from src.diagnostic import diagnose_model


def load_config(config_path):
    """加载实验配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


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
    num_clients = config['num_clients']
    rounds = config['federated_rounds']
    local_epochs = config['local_epochs']
    head_epochs = config['head_epochs']
    num_workers = config.get('num_workers', 0)

    # ==================== 阶段1: 训练 M_global ====================
    print("\n[阶段1] FedAvg 联邦训练 M_global...")
    m_global = create_model(num_classes=2, pretrained=True, device=device)
    client_loaders = create_federated_loaders(
        data_dir, num_clients=num_clients, batch_size=batch_size,
        seed=seed, num_workers=num_workers
    )
    federated_train(
        m_global, client_loaders, num_rounds=rounds,
        local_epochs=local_epochs, lr=lr, device=device
    )

    # 保存 M_global 权重
    m_global_path = os.path.join(output_dir, f'm_global_seed{seed}.pt')
    save_model(m_global, m_global_path)
    print(f"  M_global 权重已保存: {m_global_path}")

    # 保存 M_global 的 CPU 状态用于后续加载
    m_global_state = copy_model_state(m_global, device='cpu')

    # ==================== 阶段2: B 组（均衡背景 Head 重训） ====================
    print("\n[阶段2] B 组：均衡背景 Head 重训 (25/25/25/25, 冻结 backbone)")
    model_b = create_model(num_classes=2, pretrained=False, device=device)
    load_state(model_b, m_global_state)
    loader_b = create_distribution_loader(
        data_dir, 'balanced', batch_size=batch_size,
        seed=seed, num_workers=num_workers
    )
    train_head(model_b, loader_b, head_epochs, lr=lr, device=device, freeze_backbone=True)

    # 保存 B 组权重
    model_b_path = os.path.join(output_dir, f'B_seed{seed}.pt')
    save_model(model_b, model_b_path)
    print(f"  B 组权重已保存: {model_b_path}")

    # ==================== 阶段3: C 组（反相关 Head 重训） ====================
    print("\n[阶段3] C 组：反相关 Head 重训 (conflicting, 冻结 backbone)")
    model_c = create_model(num_classes=2, pretrained=False, device=device)
    load_state(model_c, m_global_state)
    loader_c = create_distribution_loader(
        data_dir, 'inverse', batch_size=batch_size,
        seed=seed, num_workers=num_workers
    )
    train_head(model_c, loader_c, head_epochs, lr=lr, device=device, freeze_backbone=True)

    # 保存 C 组权重
    model_c_path = os.path.join(output_dir, f'C_seed{seed}.pt')
    save_model(model_c, model_c_path)
    print(f"  C 组权重已保存: {model_c_path}")

    # ==================== 阶段4: 方向性诊断 ====================
    print("\n[阶段4] 方向性诊断（使用相同反事实样本评估 M_global / B / C）")
    # 准备四组均衡测试加载器（反事实配对）
    group_loaders = get_balanced_test_loader(
        data_dir, batch_size=64, num_workers=num_workers
    )

    # 诊断三个模型
    models_to_diagnose = {
        'M_global': m_global,
        'B': model_b,
        'C': model_c,
    }

    diagnostics = {}
    for tag, model in models_to_diagnose.items():
        print(f"\n  诊断 {tag}...")
        diag = diagnose_model(model, group_loaders, device)
        diagnostics[tag] = diag

        # 打印该模型的指标
        print(f"    Δ_waterbird = {diag['delta_waterbird']:+.4f}")
        print(f"    Δ_landbird  = {diag['delta_landbird']:+.4f}")
        print(f"    Original Flip Rate = {diag['original_flip_rate']*100:.2f}%")
        print(f"    Reverse Flip Rate = {diag['reverse_flip_rate']*100:.2f}%")
        ga = diag['group_accuracy']
        print(f"    Group Acc: WB-Water={ga.get('waterbird_water',0)*100:.2f}% "
              f"WB-Land={ga.get('waterbird_land',0)*100:.2f}% "
              f"LB-Land={ga.get('landbird_land',0)*100:.2f}% "
              f"LB-Water={ga.get('landbird_water',0)*100:.2f}%")

    # ==================== 阶段5: 输出该 seed 的表格 ====================
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

    return {
        'seed': seed,
        'diagnostics': diagnostics,
    }


def summarize_multi_seed(all_results):
    """多 seed 汇总：计算 Mean±Std"""
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
            values = [r['diagnostics'][tag][metric] for r in all_results if tag in r.get('diagnostics', {})]
            if values:
                summary[tag][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

        for gk in group_keys:
            values = [r['diagnostics'][tag]['group_accuracy'].get(gk, 0) for r in all_results if tag in r.get('diagnostics', {})]
            if values:
                summary[tag][f'group_acc_{gk}'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }

    # 打印汇总表
    print(f"\n{'模型':<12}{'Δ_waterbird':>20}{'Δ_landbird':>20}{'Orig Flip%':>20}{'Rev Flip%':>20}")
    print("-" * 92)
    for tag in tags:
        s = summary[tag]
        dwb = f"{s['delta_waterbird']['mean']:+.4f}±{s['delta_waterbird']['std']:.4f}"
        dlb = f"{s['delta_landbird']['mean']:+.4f}±{s['delta_landbird']['std']:.4f}"
        of = f"{s['original_flip_rate']['mean']*100:.2f}±{s['original_flip_rate']['std']*100:.2f}%"
        rf = f"{s['reverse_flip_rate']['mean']*100:.2f}±{s['reverse_flip_rate']['std']*100:.2f}%"
        print(f"{tag:<12}{dwb:>20}{dlb:>20}{of:>20}{rf:>20}")

    print(f"\n{'模型':<12}{'WB-Water':>20}{'WB-Land':>20}{'LB-Land':>20}{'LB-Water':>20}")
    print("-" * 92)
    for tag in tags:
        s = summary[tag]
        ww = f"{s['group_acc_waterbird_water']['mean']*100:.2f}±{s['group_acc_waterbird_water']['std']*100:.2f}%"
        wl = f"{s['group_acc_waterbird_land']['mean']*100:.2f}±{s['group_acc_waterbird_land']['std']*100:.2f}%"
        ll = f"{s['group_acc_landbird_land']['mean']*100:.2f}±{s['group_acc_landbird_land']['std']*100:.2f}%"
        lw = f"{s['group_acc_landbird_water']['mean']*100:.2f}±{s['group_acc_landbird_water']['std']*100:.2f}%"
        print(f"{tag:<12}{ww:>20}{wl:>20}{ll:>20}{lw:>20}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='C-Reversal 诊断实验')
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--output', type=str, default='results.json', help='结果输出 JSON 路径')
    parser.add_argument('--device', type=str, default=None, help='设备覆盖：auto/cuda/cpu')
    parser.add_argument('--smoke', action='store_true', help='冒烟测试模式（小样本+1 epoch）')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='覆盖配置中的 seeds 列表')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    if args.device:
        config['device'] = args.device
    if args.seeds:
        config['seeds'] = args.seeds

    device = get_device(config.get('device', 'auto'))
    print(f"使用设备: {device}")
    print(f"种子列表: {config['seeds']}")

    # 冒烟测试覆盖
    if args.smoke:
        config['federated_rounds'] = 1
        config['local_epochs'] = 1
        config['head_epochs'] = 1
        config['batch_size'] = 16
        config['seeds'] = [42]
        print(">>> 冒烟测试模式 (1 epoch, 1 seed, 小样本) <<<")

    # 创建输出目录
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 运行多 seed 实验
    all_results = []

    for seed in config['seeds']:
        try:
            result = run_single_seed(config, seed, device, output_dir)
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
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 多 seed 汇总
    valid_results = [r for r in all_results if 'error' not in r]
    if len(valid_results) > 0:
        summary = summarize_multi_seed(valid_results)

        # 保存完整结果（含汇总）
        final_output = {
            'per_seed': all_results,
            'summary': summary,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"\n最终结果（含汇总）已保存到: {args.output}")
        print(f"模型权重文件保存在: {output_dir}/")


if __name__ == '__main__':
    main()
