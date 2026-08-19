"""
FedAvg 联邦训练
"""
import torch
import torch.nn as nn


def local_train(model, train_loader, epochs, lr=0.001, device='cpu', freeze_backbone=False):
    """单客户端本地训练, 返回更新后的模型参数(CPU)"""
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # 返回 CPU 状态，聚合在 CPU 进行，避免设备不一致
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fedavg_aggregate(client_state_dicts):
    """FedAvg 聚合, 返回平均参数(CPU)"""
    avg_state = {}
    num_clients = len(client_state_dicts)
    for key in client_state_dicts[0].keys():
        avg_state[key] = sum(sd[key] for sd in client_state_dicts) / num_clients
    return avg_state


def federated_train(global_model, client_loaders, num_rounds, local_epochs, lr=0.001, device='cpu'):
    """完整 FedAvg 联邦训练循环, 返回训练后的全局模型"""
    from src.model import create_model, load_state

    for r in range(num_rounds):
        client_states = []
        for loader in client_loaders:
            model = create_model(num_classes=2, pretrained=False, device=device)
            load_state(model, global_model.state_dict())
            client_state = local_train(model, loader, epochs=local_epochs, lr=lr, device=device)
            client_states.append(client_state)

        global_state = fedavg_aggregate(client_states)
        load_state(global_model, global_state)
        print(f"联邦轮次 {r+1}/{num_rounds} 完成")

    return global_model
