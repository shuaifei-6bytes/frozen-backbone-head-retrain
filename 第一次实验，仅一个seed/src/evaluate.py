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
def compute_background_gap_real(model, group_loaders, device='cpu'):
    """使用真实背景替换样本计算 Background Gap 和 Flip Rate。

    group_loaders: get_balanced_test_loader 返回的四组加载器字典
    (waterbird_water / waterbird_land / landbird_land / landbird_water)。
    对水鸟比较"水背景 vs 陆地背景"下真实类别概率差，对陆鸟同理，取两者均值。
    
    同时计算 Flip Rate：统计背景切换后预测类别发生改变的比例。
    """
    model.to(device)
    model.eval()

    def collect_probs_and_preds(loader):
        probs_list = []
        preds_list = []
        y_list = []
        for images, labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            probs_list.append(probs)
            preds_list.append(preds)
            y_list.append(labels)
        return torch.cat(probs_list, 0), torch.cat(preds_list, 0), torch.cat(y_list, 0)

    # 水鸟真实类别=0：水背景 vs 陆地背景
    probs_ww, preds_ww, y1 = collect_probs_and_preds(group_loaders['waterbird_water'])
    probs_wl, preds_wl, y2 = collect_probs_and_preds(group_loaders['waterbird_land'])

    bg_gap_waterbird = 0.0
    flip_count_waterbird = 0
    total_waterbird = 0
    
    if len(y1) > 0 and len(y2) > 0:
        p_ww = probs_ww[:, 0].mean().item()
        p_wl = probs_wl[:, 0].mean().item()
        bg_gap_waterbird = abs(p_ww - p_wl)
        
        # Flip Rate: 同一批样本（按索引配对），背景切换后预测是否改变
        min_len = min(len(preds_ww), len(preds_wl))
        if min_len > 0:
            flip_count_waterbird = (preds_ww[:min_len] != preds_wl[:min_len]).sum().item()
            total_waterbird = min_len

    # 陆鸟真实类别=1：水背景 vs 陆地背景
    probs_lw, preds_lw, y3 = collect_probs_and_preds(group_loaders['landbird_water'])
    probs_ll, preds_ll, y4 = collect_probs_and_preds(group_loaders['landbird_land'])

    bg_gap_landbird = 0.0
    flip_count_landbird = 0
    total_landbird = 0
    
    if len(y3) > 0 and len(y4) > 0:
        p_lw = probs_lw[:, 1].mean().item()
        p_ll = probs_ll[:, 1].mean().item()
        bg_gap_landbird = abs(p_lw - p_ll)
        
        # Flip Rate: 同一批样本（按索引配对），背景切换后预测是否改变
        min_len = min(len(preds_lw), len(preds_ll))
        if min_len > 0:
            flip_count_landbird = (preds_lw[:min_len] != preds_ll[:min_len]).sum().item()
            total_landbird = min_len

    background_gap = (bg_gap_waterbird + bg_gap_landbird) / 2
    
    # Flip Rate: 总翻转数 / 总样本数
    total_flips = flip_count_waterbird + flip_count_landbird
    total_samples = total_waterbird + total_landbird
    flip_rate = total_flips / total_samples if total_samples > 0 else 0.0

    return background_gap, flip_rate
