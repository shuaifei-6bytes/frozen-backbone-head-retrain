"""
Head-only 重训基线（作为对照方法）。
冻结 backbone，只训练分类 head。
"""
import torch
import torch.nn as nn
import copy


def train_head_only(model, train_loader, epochs=5, lr=0.001, device='cpu'):
    """训练 head-only 基线。"""
    trained_model = copy.deepcopy(model).to(device)

    for p in trained_model.parameters():
        p.requires_grad_(False)

    # 解冻 head (fc 层)
    if hasattr(trained_model, 'fc'):
        trained_model.fc.requires_grad_(True)
    else:
        raise ValueError("模型没有 fc 属性")

    optimizer = torch.optim.Adam(
        [p for p in trained_model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    trained_model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batch = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = trained_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(f"  Head-only epoch {epoch+1}/{epochs}, loss={total_loss/max(n_batch,1):.4f}")

    return trained_model
