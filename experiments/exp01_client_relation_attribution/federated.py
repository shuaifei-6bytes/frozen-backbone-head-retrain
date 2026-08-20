"""Deterministic all-client FedAvg training."""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def _trainable_names(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def _local_update(
    global_model: nn.Module,
    loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    model = copy.deepcopy(global_model).to(device)
    model.train()
    if getattr(model, "backbone_frozen", False):
        model.backbone.eval()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    batches = 0
    for _ in range(local_epochs):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
            total_examples += int(labels.numel())
            batches += 1
    metrics = {
        "loss": total_loss / max(batches, 1),
        "accuracy": 100.0 * total_correct / max(total_examples, 1),
        "examples_seen": total_examples,
    }
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}, metrics


def _fedavg(
    model: nn.Module,
    client_states: Sequence[dict[str, torch.Tensor]],
    weights: Sequence[int],
) -> None:
    """Sample-count weighted FedAvg over trainable parameters only.

    Frozen buffers/parameters stay byte-identical to the common initialization.
    """
    trainable = _trainable_names(model)
    total_weight = float(sum(weights))
    state = model.state_dict()
    for name in trainable:
        averaged = sum(client[name].to(torch.float64) * (weight / total_weight) for client, weight in zip(client_states, weights))
        state[name] = averaged.to(dtype=state[name].dtype)
    model.load_state_dict(state, strict=True)


def train_federated(
    model: nn.Module,
    client_loaders: Sequence[DataLoader],
    rounds: int,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    model.to(device)
    history: list[dict[str, Any]] = []
    sample_weights = [len(loader.dataset) for loader in client_loaders]
    for round_index in range(rounds):
        client_states = []
        client_metrics = []
        for client_id, loader in enumerate(client_loaders):
            state, metrics = _local_update(model, loader, local_epochs, learning_rate, device)
            client_states.append(state)
            client_metrics.append({"client_id": client_id, **metrics})
        _fedavg(model, client_states, sample_weights)
        row = {
            "round": round_index + 1,
            "mean_loss": float(np.mean([item["loss"] for item in client_metrics])),
            "mean_accuracy": float(np.mean([item["accuracy"] for item in client_metrics])),
            "clients": client_metrics,
        }
        history.append(row)
        logger.info(
            "round %d/%d | loss=%.6f | mean_client_accuracy=%.2f%%",
            round_index + 1,
            rounds,
            row["mean_loss"],
            row["mean_accuracy"],
        )
    return history
