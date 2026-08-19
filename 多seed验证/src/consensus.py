"""
跨 minibatch 共识聚合。
"""
import torch
import numpy as np


def aggregate_consensus(mb_scores):
    """跨 minibatch 共识聚合。

    输入：mb_scores: list[dict]，每个 dict 是 layer_name -> Tensor
    输出：list of (layer_name, channel_idx, combined_score)，按分数降序
    """
    if not mb_scores:
        return []

    all_layers = mb_scores[0].keys()
    num_mb = len(mb_scores)
    all_units = []

    for layer_name in all_layers:
        scores_list = []
        for mb in mb_scores:
            if layer_name in mb:
                scores_list.append(mb[layer_name].detach().cpu().numpy())
        if not scores_list:
            continue

        scores_array = np.stack(scores_list, axis=0)  # [num_mb, C_out]
        importance = scores_array.mean(axis=0)

        # 频率：该通道在多少 minibatch 中排名 top-20%
        top_k = max(1, scores_array.shape[1] // 5)
        frequency = np.zeros(scores_array.shape[1])
        for mb_idx in range(num_mb):
            top_indices = np.argsort(scores_array[mb_idx])[-top_k:]
            frequency[top_indices] += 1
        frequency = frequency / num_mb

        # 稳定性：分数标准差的倒数
        std = scores_array.std(axis=0)
        stability = 1.0 / (std + 1e-6)

        # 归一化
        imp_norm = importance / (importance.max() + 1e-6)
        freq_norm = frequency / (frequency.max() + 1e-6)
        stab_norm = stability / (stability.max() + 1e-6)

        combined = 0.5 * imp_norm + 0.3 * freq_norm + 0.2 * stab_norm

        for ch_idx in range(len(combined)):
            all_units.append((layer_name, ch_idx, float(combined[ch_idx])))

    all_units.sort(key=lambda x: x[2], reverse=True)
    return all_units
