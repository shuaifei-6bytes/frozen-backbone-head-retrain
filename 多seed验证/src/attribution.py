"""
Minibatch 级别 backbone 功能单元归因。
候选单元：ResNet-18 各卷积层通道。
归因方式：基于梯度的通道重要性（在 relation-conflicting 样本上计算）。
"""
import torch
import torch.nn as nn


def compute_minibatch_attribution(model, dataloader, device='cpu'):
    """对一个 minibatch 计算各卷积通道的 relation 归因分数。

    原理：在 conflicting 样本上计算 loss 对每个 conv 输出梯度的 L2 范数
    （沿通道维度平均），作为通道重要性。

    返回：
        scores: dict[str, Tensor]，每个 layer 一个长度为 out_channels 的 tensor
    """
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)

    grads = {}

    def make_hook(name):
        def hook_fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                g = grad_output[0].abs().mean(dim=(0, 2, 3))
                grads[name] = g
        return hook_fn

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_full_backward_hook(make_hook(name)))

    criterion = nn.CrossEntropyLoss()
    for images, labels, places in dataloader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad_(True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

    for h in handles:
        h.remove()

    scores = {}
    for name, g in grads.items():
        # 确保梯度张量在正确设备上
        g = g.to(device)
        if g.sum() > 0:
            scores[name] = g / g.sum()
        else:
            scores[name] = g
    return scores


def compute_all_minibatches_attribution(model, dataloader, num_minibatches, device='cpu'):
    """将 dataloader 切成 num_minibatches 份，分别计算归因分数。

    返回：
        mb_scores: list[dict]，每个 dict 是 layer->scores
    """
    all_images, all_labels, all_places = [], [], []
    for images, labels, places in dataloader:
        all_images.append(images)
        all_labels.append(labels)
        all_places.append(places)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_places = torch.cat(all_places, dim=0)

    total = all_images.size(0)
    mb_size = max(1, total // num_minibatches)

    mb_scores = []
    for m in range(num_minibatches):
        start = m * mb_size
        end = min((m + 1) * mb_size, total)
        if start >= end:
            break
        sub_ds = torch.utils.data.TensorDataset(
            all_images[start:end], all_labels[start:end], all_places[start:end])
        sub_loader = torch.utils.data.DataLoader(sub_ds, batch_size=16, shuffle=False)
        scores = compute_minibatch_attribution(model, sub_loader, device=device)
        mb_scores.append(scores)
        print(f"  minibatch {m+1}/{num_minibatches} 样本数={end-start}")

    return mb_scores
