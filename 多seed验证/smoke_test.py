"""
极简冒烟测试：验证多seed验证实验的完整pipeline可以跑通
用极少量样本和1 epoch验证代码无报错
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np

from src.model import create_model, get_device
from src.data import WaterbirdsDataset, get_transforms

print("=" * 60)
print("多Seed验证实验 - 极简冒烟测试")
print("=" * 60)

device = 'cpu'
data_dir = "/mnt/d/VS Code 项目/federated_relation_unlearning/新建文件夹/冻结backbone+Head重训/waterbird_complete95_forest2water2"

# 创建小数据集加载器
print("\n[1] 创建小数据集...")
train_ds = WaterbirdsDataset(data_dir, split='train', transform=get_transforms('train'))
test_ds = WaterbirdsDataset(data_dir, split='test', transform=get_transforms('val'))

# 限制样本数
max_samples = 100
train_ds.samples = train_ds.samples[:max_samples]
test_ds.samples = test_ds.samples[:max_samples]

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False)
print(f"  训练样本: {len(train_ds)}, 测试样本: {len(test_ds)}")

# 创建模型
print("\n[2] 创建 ResNet-18 模型...")
device_obj = torch.device(device)
model = create_model(num_classes=2, pretrained=False, device=device).to(device_obj)
print(f"  模型创建完成")

# 评估模型
print("\n[3] 评估模型...")
from src.evaluate import evaluate_model, compute_background_gap_real
oa, wg, ga = evaluate_model(model, test_loader, device)
bg = compute_background_gap_real(model, test_loader, device)
print(f"  Overall Acc: {oa:.4f}, Worst-group Acc: {wg:.4f}, BG: {bg:.4f}")

# 归因测试
print("\n[4] 测试归因模块...")
from src.attribution import compute_minibatch_attribution
scores = compute_minibatch_attribution(model, train_loader, device)
total_channels = sum(len(v) for v in scores.values())
print(f"  归因完成，共 {len(scores)} 层, {total_channels} 个通道")

# 共识聚合
print("\n[5] 测试共识聚合...")
from src.consensus import aggregate_consensus
mb_scores = [scores, scores, scores]  # 模拟3个minibatch
consensus = aggregate_consensus(mb_scores)
top_5 = consensus[:5]
print(f"  Top-5 通道:")
for i, (layer, ch, sc) in enumerate(top_5):
    print(f"    {i+1}. {layer} ch{ch} score={sc:.4f}")

# 稀疏编辑测试
print("\n[6] 测试稀疏编辑...")
from src.sparse_edit import apply_sparse_edit
top_3 = consensus[:3]
edited_model = apply_sparse_edit(model, top_3, train_loader, epochs=1, lr=0.001, device=device)
print(f"  稀疏编辑完成")

# Head-only 测试
print("\n[7] 测试 Head-only...")
from src.head_only import train_head_only
head_model = train_head_only(model, train_loader, epochs=1, lr=0.001, device=device)
print(f"  Head-only 训练完成")

# 验证门控测试
print("\n[8] 测试验证门控...")
from src.validation_gate import validate_candidate
passed, metrics = validate_candidate(
    model, edited_model, test_loader, test_loader, device,
    tau_D=0.01, tau_P=0.05
)
print(f"  门控结果: passed={passed}")
print(f"  指标: {metrics}")

print("\n" + "=" * 60)
print("✓ 冒烟测试全部通过！Pipeline 正常工作。")
print("=" * 60)
print("""
下一步：
1. 先用父目录的 run_experiment.py 训练 M_global
2. 将 m_global.pt 路径填入 config.json
3. 在 Kaggle GPU 上运行完整的 run_multi_seed.py
""")
