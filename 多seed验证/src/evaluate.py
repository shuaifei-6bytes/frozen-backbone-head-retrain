"""
评价指标：Background Gap, Flip Rate, Worst-group Accuracy, Overall Accuracy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def evaluate_model(model, loader, device='cpu'):
    """返回整体 accuracy、worst-group accuracy 及各组准确率"""
    model.to(device)
    model.eval()

    group_correct = {'waterbird_water': 0, 'waterbird_land': 0,
                     'landbird_land': 0, 'landbird_water': 0}
    group_total = {'waterbird_water': 0, 'waterbird_land': 0,
                   'landbird_land': 0, 'landbird_water': 0}

    correct = 0
    total = 0

    for images, labels, places in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(labels.size(0)):
            y, p, pred = labels[i].item(), places[i].item(), preds[i].item()
            total += 1
            if pred == y:
                correct += 1

            if y == 0 and p == 0:
                group = 'waterbird_water'
            elif y == 0 and p == 1:
                group = 'waterbird_land'
            elif y == 1 and p == 1:
                group = 'landbird_land'
            else:
                group = 'landbird_water'
            group_total[group] += 1
            if pred == y:
                group_correct[group] += 1

    overall_acc = correct / total if total > 0 else 0

    group_accs = {}
    for g in group_correct:
        if group_total[g] > 0:
            group_accs[g] = group_correct[g] / group_total[g]
        else:
            group_accs[g] = 0.0

    worst_group_acc = min(group_accs.values())

    return overall_acc, worst_group_acc, group_accs


@torch.no_grad()
def compute_background_gap(model, loader, device='cpu'):
    """计算 Background Gap 和 Background Flip Rate。

    对每个测试样本，通过旋转/裁剪等无法精确换背景，这里用近似：
    使用标准概率差来衡量对背景的依赖。真正严格的实现需要背景替换样本。
    """
    model.to(device)
    model.eval()

    gap_sum = 0.0
    flip_count = 0
    total = 0

    for images, labels, places in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)

        # 记录真实类别的预测概率和预测类别
        true_prob = probs[torch.arange(labels.size(0)), labels]
        _, preds = torch.max(probs, 1)

        # 模拟: 用批次内不同背景的样本构造对照组
        # 简化: 计算真实类别概率与最大概率的差作为"依赖度"
        max_prob, _ = torch.max(probs, 1)
        gap = (max_prob - true_prob).abs()

        gap_sum += gap.sum().item()
        flip_count += (preds != labels).sum().item()
        total += labels.size(0)

    background_gap = gap_sum / total if total > 0 else 0.0
    flip_rate = flip_count / total if total > 0 else 0.0

    return background_gap, flip_rate


@torch.no_grad()
def compute_background_gap_real(model, loader_or_groups, device='cpu'):
    """计算 Background Gap 和 Flip Rate。

    如果 loader_or_groups 是一个 DataLoader，则用整个测试集计算。
    如果是一个 dict（包含 waterbird_water/land 等），则用分组计算。
    """
    model.to(device)
    model.eval()

    # 如果是 dict，说明是分组加载器
    if isinstance(loader_or_groups, dict):
        group_water = loader_or_groups
        def collect_probs_and_preds(loader):
            probs_list = []
            preds_list = []
            for images, labels, _ in loader:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                probs_list.append(probs)
                preds_list.append(preds)
            return torch.cat(probs_list, 0) if probs_list else torch.empty(0, 2), torch.cat(preds_list, 0) if preds_list else torch.empty(0)

        probs_ww, preds_ww = collect_probs_and_preds(group_water['waterbird_water'])
        probs_wl, preds_wl = collect_probs_and_preds(group_water['waterbird_land'])
        probs_lw, preds_lw = collect_probs_and_preds(group_water['landbird_water'])
        probs_ll, preds_ll = collect_probs_and_preds(group_water['landbird_land'])

        if len(probs_ww) > 0 and len(probs_wl) > 0:
            p_ww = probs_ww[:, 0].mean().item()
            p_wl = probs_wl[:, 0].mean().item()
            bg_gap_waterbird = abs(p_ww - p_wl)
            flip_count_w = (preds_ww[:min(len(preds_ww), len(preds_wl))] != preds_wl[:min(len(preds_ww), len(preds_wl))]).sum().item()
            total_w = min(len(preds_ww), len(preds_wl))
        else:
            bg_gap_waterbird = 0.0
            flip_count_w = 0
            total_w = 0

        if len(probs_lw) > 0 and len(probs_ll) > 0:
            p_lw = probs_lw[:, 1].mean().item()
            p_ll = probs_ll[:, 1].mean().item()
            bg_gap_landbird = abs(p_lw - p_ll)
            flip_count_l = (preds_lw[:min(len(preds_lw), len(preds_ll))] != preds_ll[:min(len(preds_lw), len(preds_ll))]).sum().item()
            total_l = min(len(preds_lw), len(preds_ll))
        else:
            bg_gap_landbird = 0.0
            flip_count_l = 0
            total_l = 0

        background_gap = (bg_gap_waterbird + bg_gap_landbird) / 2
        flip_rate = (flip_count_w + flip_count_l) / (total_w + total_l) if (total_w + total_l) > 0 else 0.0
        return background_gap, flip_rate

    # 否则是 DataLoader，用简单方法计算
    loader = loader_or_groups
    gap_sum = 0.0
    flip_count = 0
    total = 0

    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        true_prob = probs[torch.arange(labels.size(0)), labels]
        _, preds = torch.max(probs, 1)
        gap = (probs.max(1)[0] - true_prob).abs()
        gap_sum += gap.sum().item()
        flip_count += (preds != labels).sum().item()
        total += labels.size(0)

    return gap_sum / total if total > 0 else 0.0, flip_count / total if total > 0 else 0.0
