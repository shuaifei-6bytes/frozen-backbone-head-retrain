"""
稀疏编辑：只修改选中的 top-K 功能单元（通道）。
"""
import torch
import torch.nn as nn
import copy


def apply_sparse_edit(model, selected_units, dataloader, device='cpu', epochs=5, lr=0.001):
    """对选中的功能单元进行稀疏编辑。

    输入：
        selected_units: list of (layer_name, channel_idx, score)
    """
    edited_model = copy.deepcopy(model).to(device)

    # 冻结所有参数
    for p in edited_model.parameters():
        p.requires_grad_(False)

    # 解冻选中通道的权重
    selected_set = set((layer, ch) for layer, ch, _ in selected_units)
    for name, module in edited_model.named_modules():
        if isinstance(module, nn.Conv2d):
            channels_to_edit = [ch for l, ch in selected_set if l == name]
            if channels_to_edit:
                module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    # 只训练可编辑参数
    params = [p for p in edited_model.parameters() if p.requires_grad]
    if not params:
        print("  警告：无可编辑参数")
        return edited_model

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    edited_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = edited_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  稀疏编辑 epoch {epoch+1}/{epochs}, loss={total_loss/max(n_batch,1):.4f}")

    return edited_model
