"""
C-Reversal 诊断：反相关 Head 重训是否产生反向背景依赖

核心指标：
1. Signed Background Effect (Δ_waterbird, Δ_landbird) — 保留背景影响的方向
2. Directional Flip Rate (Original-direction / Reverse-direction)
3. 四种 Group Accuracy

依据 writing-block(1).md 设计，不重训任何模型，只加载已保存权重做诊断。
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


@torch.no_grad()
def collect_probs_preds(model, loader, device='cpu'):
    """收集模型在给定 loader 上的 softmax 概率和预测类别"""
    model.to(device)
    model.eval()

    probs_list = []
    preds_list = []
    labels_list = []

    for images, labels, _ in loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(probs, 1)

        probs_list.append(probs.cpu())
        preds_list.append(preds.cpu())
        labels_list.append(labels)

    if not probs_list:
        return torch.empty(0, 2), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    return torch.cat(probs_list, 0), torch.cat(preds_list, 0), torch.cat(labels_list, 0)


@torch.no_grad()
def compute_signed_background_effect(model, group_loaders, device='cpu'):
    """计算有符号背景效应（核心指标一）

    Δ_waterbird = P(水鸟|水鸟主体+水背景) - P(水鸟|水鸟主体+陆地背景)
    Δ_landbird  = P(陆鸟|陆鸟主体+陆地背景) - P(陆鸟|陆鸟主体+水背景)

    返回值:
      Δ_waterbird > 0: 水背景促进预测水鸟（原方向依赖）
      Δ_waterbird ≈ 0: 背景无影响（真正去关联）
      Δ_waterbird < 0: 陆地背景反而促进预测水鸟（反向依赖）
    同理 Δ_landbird。
    """
    # 水鸟：真实类别=0，比较水背景 vs 陆地背景
    probs_ww, _, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)

    delta_waterbird = 0.0
    if len(probs_ww) > 0 and len(probs_wl) > 0:
        p_waterbird_given_water_bg = probs_ww[:, 0].mean().item()
        p_waterbird_given_land_bg = probs_wl[:, 0].mean().item()
        delta_waterbird = p_waterbird_given_water_bg - p_waterbird_given_land_bg

    # 陆鸟：真实类别=1，比较陆地背景 vs 水背景
    probs_ll, _, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)

    delta_landbird = 0.0
    if len(probs_ll) > 0 and len(probs_lw) > 0:
        p_landbird_given_land_bg = probs_ll[:, 1].mean().item()
        p_landbird_given_water_bg = probs_lw[:, 1].mean().item()
        delta_landbird = p_landbird_given_land_bg - p_landbird_given_water_bg

    return delta_waterbird, delta_landbird


@torch.no_grad()
def compute_directional_flip_rate(model, group_loaders, device='cpu'):
    """计算方向性翻转率（核心指标二）

    Original-direction Flip: 换背景后预测按原始伪关联方向翻转
      例如水鸟：水背景→预测水鸟，换陆地背景→预测陆鸟
    Reverse-direction Flip: 换背景后预测按反方向翻转
      例如水鸟：水背景→预测陆鸟，换陆地背景→预测水鸟

    水鸟配对：(waterbird_water, waterbird_land)
    陆鸟配对：(landbird_land, landbird_water)
    """
    # 水鸟：水背景 vs 陆地背景
    _, preds_ww, y_ww = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    _, preds_wl, y_wl = collect_probs_preds(model, group_loaders['waterbird_land'], device)

    # 陆鸟：陆地背景 vs 水背景
    _, preds_ll, y_ll = collect_probs_preds(model, group_loaders['landbird_land'], device)
    _, preds_lw, y_lw = collect_probs_preds(model, group_loaders['landbird_water'], device)

    original_flips = 0
    reverse_flips = 0
    total_pairs = 0

    # 水鸟配对（按索引配对，取最小长度）
    min_w = min(len(preds_ww), len(preds_wl))
    for i in range(min_w):
        pred_water_bg = preds_ww[i].item()  # 水背景下的预测
        pred_land_bg = preds_wl[i].item()   # 陆地背景下的预测
        total_pairs += 1

        if pred_water_bg != pred_land_bg:
            # 发生了翻转，判断方向
            # 原始方向：水背景→水鸟(0)，陆地背景→陆鸟(1)
            if pred_water_bg == 0 and pred_land_bg == 1:
                original_flips += 1
            # 反向：水背景→陆鸟(1)，陆地背景→水鸟(0)
            elif pred_water_bg == 1 and pred_land_bg == 0:
                reverse_flips += 1

    # 陆鸟配对
    min_l = min(len(preds_ll), len(preds_lw))
    for i in range(min_l):
        pred_land_bg = preds_ll[i].item()   # 陆地背景下的预测
        pred_water_bg = preds_lw[i].item()  # 水背景下的预测
        total_pairs += 1

        if pred_water_bg != pred_land_bg:
            # 原始方向：陆地背景→陆鸟(1)，水背景→水鸟(0)
            if pred_land_bg == 1 and pred_water_bg == 0:
                original_flips += 1
            # 反向：陆地背景→水鸟(0)，水背景→陆鸟(1)
            elif pred_land_bg == 0 and pred_water_bg == 1:
                reverse_flips += 1

    original_flip_rate = original_flips / total_pairs if total_pairs > 0 else 0.0
    reverse_flip_rate = reverse_flips / total_pairs if total_pairs > 0 else 0.0

    return original_flip_rate, reverse_flip_rate


@torch.no_grad()
def compute_group_accuracies(model, group_loaders, device='cpu'):
    """计算四种 Group Accuracy（辅助指标）"""
    model.to(device)
    model.eval()

    group_accs = {}
    for name, loader in group_loaders.items():
        correct = 0
        total = 0
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        group_accs[name] = correct / total if total > 0 else 0.0

    return group_accs


@torch.no_grad()
def compute_per_sample_signed_effect(model, group_loaders, device='cpu'):
    """保留每个样本的 signed effect（执行约束第8条）

    返回两个列表：waterbird 样本和 landbird 样本的有符号效应
    """
    # 水鸟
    probs_ww, _, _ = collect_probs_preds(model, group_loaders['waterbird_water'], device)
    probs_wl, _, _ = collect_probs_preds(model, group_loaders['waterbird_land'], device)

    waterbird_effects = []
    min_w = min(len(probs_ww), len(probs_wl))
    for i in range(min_w):
        p_w_bg = probs_ww[i, 0].item()  # 水背景下预测水鸟的概率
        p_l_bg = probs_wl[i, 0].item()  # 陆地背景下预测水鸟的概率
        waterbird_effects.append(p_w_bg - p_l_bg)

    # 陆鸟
    probs_ll, _, _ = collect_probs_preds(model, group_loaders['landbird_land'], device)
    probs_lw, _, _ = collect_probs_preds(model, group_loaders['landbird_water'], device)

    landbird_effects = []
    min_l = min(len(probs_ll), len(probs_lw))
    for i in range(min_l):
        p_l_bg = probs_ll[i, 1].item()  # 陆地背景下预测陆鸟的概率
        p_w_bg = probs_lw[i, 1].item()  # 水背景下预测陆鸟的概率
        landbird_effects.append(p_l_bg - p_w_bg)

    return waterbird_effects, landbird_effects


@torch.no_grad()
def diagnose_model(model, group_loaders, device='cpu'):
    """对单个模型做完整诊断，返回所有指标"""
    # 核心指标一：有符号背景效应
    delta_wb, delta_lb = compute_signed_background_effect(model, group_loaders, device)

    # 核心指标二：方向性翻转率
    orig_flip, rev_flip = compute_directional_flip_rate(model, group_loaders, device)

    # 辅助指标：四种 Group Accuracy
    group_accs = compute_group_accuracies(model, group_loaders, device)

    # 每样本 signed effect（约束第8条）
    wb_effects, lb_effects = compute_per_sample_signed_effect(model, group_loaders, device)

    return {
        'delta_waterbird': float(delta_wb),
        'delta_landbird': float(delta_lb),
        'original_flip_rate': float(orig_flip),
        'reverse_flip_rate': float(rev_flip),
        'group_accuracy': {k: float(v) for k, v in group_accs.items()},
        'per_sample': {
            'waterbird_effects': [float(x) for x in wb_effects],
            'landbird_effects': [float(x) for x in lb_effects],
        }
    }
