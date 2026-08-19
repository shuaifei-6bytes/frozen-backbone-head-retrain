"""
Waterbirds 联邦伪关联 Head-only 可行性实验 V1
主入口：5-client IID FedAvg 训练 M_global，再跑 A/B/C/D 对照，输出四项指标。
用法:
  python run_experiment.py                 # 完整实验
  python run_experiment.py --smoke         # 1 epoch 冒烟测试 (自动用 CPU/小样本)
  python run_experiment.py --device cuda   # 用 GPU 跑
"""
import argparse
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import get_test_loader, get_balanced_test_loader, create_federated_loaders
from src.model import create_model, copy_model_state, get_device
from src.federated import federated_train
from src.train_head import create_distribution_loader, train_head
from src.evaluate import evaluate_model, compute_background_gap_real


ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'waterbird_complete95_forest2water2')


def run_eval(model, device, tag, max_test=None, workers=0):
    """评估模型并打印四项指标"""
    test_loader = get_test_loader(ROOT_DIR, batch_size=64, max_samples=max_test, num_workers=workers)
    group_loaders = get_balanced_test_loader(ROOT_DIR, batch_size=64, max_samples_per_group=max_test, num_workers=workers)

    overall_acc, worst_acc, group_accs = evaluate_model(model, test_loader, device)
    bg_gap, flip_rate = compute_background_gap_real(model, group_loaders, device)

    print(f"\n=== {tag} 指标 ===")
    print(f"  Overall Accuracy:      {overall_acc*100:.2f}%")
    print(f"  Worst-group Accuracy:  {worst_acc*100:.2f}%")
    print(f"  Background Gap:        {bg_gap:.4f}")
    print(f"  Flip Rate:             {flip_rate*100:.2f}%")

    return overall_acc, worst_acc, bg_gap, flip_rate, group_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true', help='冒烟测试 (自动小样本+1 epoch)')
    parser.add_argument('--device', type=str, default='auto', help='设备：auto/cuda/cpu')
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=30, help='联邦聚合轮次 (完整实验=30)')
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--head_epochs', type=int, default=20, help='Head 重训 epochs (完整=20)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42, help='单 seed 模式下的随机种子')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='多 seed 列表，如 --seeds 42 123 456 789 1024')
    parser.add_argument('--output', type=str, default='results.json', help='结果输出 JSON 路径（多 seed 模式生效）')
    parser.add_argument('--pretrained', action='store_true', default=True, help='用 ImageNet 预训练权重')
    parser.add_argument('--max_train', type=int, default=None, help='冒烟时限制每客户端训练样本数')
    parser.add_argument('--max_test', type=int, default=None, help='冒烟时限制测试样本数/组')
    parser.add_argument('--workers', type=int, default=0, help='DataLoader worker 数 (Kaggle 建议 4)')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"使用设备：{device}")

    if args.smoke:
        args.rounds = 1
        args.local_epochs = 1
        args.head_epochs = 1
        args.batch_size = 16
        args.pretrained = False
        if args.max_train is None:
            args.max_train = 100
        if args.max_test is None:
            args.max_test = 50
        print(">>> 冒烟测试模式 (1 epoch, 小样本) <<<")

    torch.manual_seed(args.seed)

    # 决定 seed 列表
    seed_list = args.seeds if args.seeds else [args.seed]
    if args.seeds:
        print(f"多 seed 模式：{seed_list}")
        print(f"结果将保存到：{args.output}")

    # ========== 多 seed 循环 ==========
    all_results = []

    for current_seed in seed_list:
        try:
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            print(f"\n{'='*60}")
            print(f"Seed {current_seed}")
            print(f"{'='*60}")

            # ========== 阶段 1: FedAvg 训练 M_global ==========
            print("\n[阶段 1] FedAvg 联邦训练 M_global...")
            global_model = create_model(num_classes=2, pretrained=args.pretrained, device=device)
            client_loaders = create_federated_loaders(ROOT_DIR, num_clients=args.num_clients,
                                                      batch_size=args.batch_size, seed=current_seed,
                                                      max_samples=args.max_train, num_workers=args.workers)
            federated_train(global_model, client_loaders, num_rounds=args.rounds,
                            local_epochs=args.local_epochs, lr=args.lr, device=device)

            # 保存 M_global 状态 (CPU)
            m_global_state = copy_model_state(global_model, device='cpu')

            # 评估 M_global
            m_global_results = run_eval(global_model, device, "M_global (基准)", max_test=args.max_test, workers=args.workers)

            # ========== 阶段 2: A/B/C/D 对照 ==========
            results = {'M_global': m_global_results}

            # A 组：原分布 Head 重训 (95/5, 冻结 backbone)
            print("\n[A 组] 原分布 Head 重训 (95/5, 冻结 backbone)")
            model_a = create_model(num_classes=2, pretrained=False, device=device)
            model_a.load_state_dict({k: v.to(device) for k, v in m_global_state.items()})
            loader_a = create_distribution_loader(ROOT_DIR, 'original', batch_size=args.batch_size,
                                                  seed=current_seed, max_samples=args.max_train, num_workers=args.workers)
            train_head(model_a, loader_a, args.head_epochs, lr=args.lr, device=device, freeze_backbone=True)
            results['A'] = run_eval(model_a, device, "A 组 (原分布 Head 重训)", max_test=args.max_test, workers=args.workers)

            # B 组：均衡背景 Head 重训 (25/25/25/25, 冻结 backbone)
            print("\n[B 组] 均衡背景 Head 重训 (25/25/25/25, 冻结 backbone)")
            model_b = create_model(num_classes=2, pretrained=False, device=device)
            model_b.load_state_dict({k: v.to(device) for k, v in m_global_state.items()})
            loader_b = create_distribution_loader(ROOT_DIR, 'balanced', batch_size=args.batch_size,
                                                  seed=current_seed, max_samples=args.max_train, num_workers=args.workers)
            train_head(model_b, loader_b, args.head_epochs, lr=args.lr, device=device, freeze_backbone=True)
            results['B'] = run_eval(model_b, device, "B 组 (均衡背景 Head 重训)", max_test=args.max_test, workers=args.workers)

            # C 组：反相关 Head 重训 (conflicting, 冻结 backbone)
            print("\n[C 组] 反相关 Head 重训 (conflicting, 冻结 backbone)")
            model_c = create_model(num_classes=2, pretrained=False, device=device)
            model_c.load_state_dict({k: v.to(device) for k, v in m_global_state.items()})
            loader_c = create_distribution_loader(ROOT_DIR, 'inverse', batch_size=args.batch_size,
                                                  seed=current_seed, max_samples=args.max_train, num_workers=args.workers)
            train_head(model_c, loader_c, args.head_epochs, lr=args.lr, device=device, freeze_backbone=True)
            results['C'] = run_eval(model_c, device, "C 组 (反相关 Head 重训)", max_test=args.max_test, workers=args.workers)

            # D 组：正常继续训练 (不冻结，95/5)
            print("\n[D 组] 正常继续训练 Baseline (全参数更新，95/5)")
            model_d = create_model(num_classes=2, pretrained=False, device=device)
            model_d.load_state_dict({k: v.to(device) for k, v in m_global_state.items()})
            loader_d = create_distribution_loader(ROOT_DIR, 'original', batch_size=args.batch_size,
                                                  seed=current_seed, max_samples=args.max_train, num_workers=args.workers)
            train_head(model_d, loader_d, args.head_epochs, lr=args.lr, device=device, freeze_backbone=False)
            results['D'] = run_eval(model_d, device, "D 组 (正常继续训练)", max_test=args.max_test, workers=args.workers)

            # ========== 结果汇总 ==========
            print("\n\n========== 实验结果汇总 ==========")
            print(f"{'组':<15}{'Overall%':>12}{'Worst%':>12}{'BG Gap':>12}{'Flip%':>12}")
            print("-" * 64)
            for tag, (oa, wa, bg, fr, _) in results.items():
                print(f"{tag:<15}{oa*100:>12.2f}{wa*100:>12.2f}{bg:>12.4f}{fr*100:>12.2f}")

            # 成功判定（相对 M_global）
            oa_base, wa_base, bg_base, fr_base, _ = results['M_global']
            oa_b, wa_b, bg_b, fr_b, _ = results['B']
            print("\n=== B 组 判定 (相对 M_global) ===")
            bg_drop = ((bg_base - bg_b) / bg_base * 100) if bg_base > 0 else 0
            fr_drop = ((fr_base - fr_b) / fr_base * 100) if fr_base > 0 else 0
            wg_gain = (wa_b - wa_base) * 100
            oa_change = (oa_b - oa_base) * 100
            print(f"  Background Gap 下降：{bg_drop:.1f}% (目标≥50%)")
            print(f"  Flip Rate 下降：{fr_drop:.1f}% (目标≥50%)")
            print(f"  Worst-group Acc 提升：{wg_gain:.2f}pp (目标≥10pp)")
            print(f"  Overall Acc 变化：{oa_change:.2f}pp (下降≤2pp)")
            success = (bg_drop > 50 if bg_base > 0 else False) and (fr_drop > 50 if fr_base > 0 else False) and wg_gain >= 10 and oa_change >= -2
            print(f"  成功判定：{'通过 ✓' if success else '未通过 ✗'}")

            # 保存单 seed 结果
            seed_result = {
                'seed': current_seed,
                'results': results,
                'success': success,
                'bg_drop': bg_drop,
                'fr_drop': fr_drop,
                'wg_gain': wg_gain,
                'oa_change': oa_change,
            }
            all_results.append(seed_result)

            # 多 seed 模式下每完成一个 seed 就保存一次
            if args.seeds:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"\n✓ Seed {current_seed} 完成，结果已保存到 {args.output}")

        except Exception as e:
            print(f"\n✗ Seed {current_seed} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'seed': current_seed,
                'error': str(e),
            })
            if args.seeds:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ========== 多 seed 汇总 ==========
    if args.seeds and len(all_results) > 1:
        print(f"\n{'='*60}")
        print("多 Seed 实验汇总")
        print(f"{'='*60}")

        success_count = sum(1 for r in all_results if r.get('success', False))
        print(f"成功 seed 数: {success_count}/{len(all_results)}")

        if success_count > 0:
            print("\n各指标统计:")
            bg_drops = [r['bg_drop'] for r in all_results if 'bg_drop' in r]
            fr_drops = [r['fr_drop'] for r in all_results if 'fr_drop' in r]
            wg_gains = [r['wg_gain'] for r in all_results if 'wg_gain' in r]
            oa_changes = [r['oa_change'] for r in all_results if 'oa_change' in r]

            print(f"  Background Gap 下降: {np.mean(bg_drops):.1f}% ± {np.std(bg_drops):.1f}%")
            print(f"  Flip Rate 下降: {np.mean(fr_drops):.1f}% ± {np.std(fr_drops):.1f}%")
            print(f"  Worst-group Acc 提升: {np.mean(wg_gains):.2f}pp ± {np.std(wg_gains):.2f}pp")
            print(f"  Overall Acc 变化: {np.mean(oa_changes):.2f}pp ± {np.std(oa_changes):.2f}pp")

        print(f"\n详细结果已保存到: {args.output}")

    # 单 seed 模式下也保存结果
    if not args.seeds and len(all_results) == 1:
        output_path = args.output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results[0], f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
