"""Natural-test accuracy and strict same-subject counterfactual audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .config import GROUPS


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    correct = {group: 0 for group in GROUPS}
    counts = {group: 0 for group in GROUPS}
    all_correct = 0
    all_count = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        matches = predictions.eq(labels)
        all_correct += int(matches.sum().item())
        all_count += int(labels.numel())
        for index, group in enumerate(batch["group"]):
            counts[group] += 1
            correct[group] += int(matches[index].item())
    if all_count == 0 or any(counts[group] == 0 for group in GROUPS):
        raise ValueError("Natural test set must contain samples from all four groups")
    group_accuracies = {group: 100.0 * correct[group] / counts[group] for group in GROUPS}
    return {
        "overall_accuracy": 100.0 * all_correct / all_count,
        "worst_group_accuracy": min(group_accuracies.values()),
        "group_accuracies": group_accuracies,
        "group_counts": counts,
    }


@torch.no_grad()
def evaluate_counterfactual(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    labels: list[int] = []
    subject_ids: list[str] = []
    p_water_background: list[float] = []
    p_land_background: list[float] = []
    pred_water_background: list[int] = []
    pred_land_background: list[int] = []
    for batch in loader:
        water = batch["water"].to(device, non_blocking=True)
        land = batch["land"].to(device, non_blocking=True)
        y = batch["label"].to(device)
        water_prob = F.softmax(model(water), dim=1)
        land_prob = F.softmax(model(land), dim=1)
        target_water_prob = torch.where(y.eq(1), water_prob[:, 1], water_prob[:, 0])
        target_land_prob = torch.where(y.eq(1), land_prob[:, 1], land_prob[:, 0])
        labels.extend(int(value) for value in y.cpu().tolist())
        subject_ids.extend(str(value) for value in batch["subject_id"])
        p_water_background.extend(float(value) for value in target_water_prob.cpu().tolist())
        p_land_background.extend(float(value) for value in target_land_prob.cpu().tolist())
        pred_water_background.extend(int(value) for value in water_prob.argmax(dim=1).cpu().tolist())
        pred_land_background.extend(int(value) for value in land_prob.argmax(dim=1).cpu().tolist())

    y = np.asarray(labels, dtype=np.int64)
    p_water = np.asarray(p_water_background, dtype=np.float64)
    p_land = np.asarray(p_land_background, dtype=np.float64)
    pred_water = np.asarray(pred_water_background, dtype=np.int64)
    pred_land = np.asarray(pred_land_background, dtype=np.int64)
    waterbird = y == 1
    landbird = y == 0
    if not waterbird.any() or not landbird.any():
        raise ValueError("Counterfactual audit requires both subject classes")

    # Positive means the aligned background increases the correct-class probability.
    delta_waterbird = float(np.mean(p_water[waterbird] - p_land[waterbird]))
    delta_landbird = float(np.mean(p_land[landbird] - p_water[landbird]))
    background_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2.0

    original = ((y == 1) & (pred_water == 1) & (pred_land == 0)) | (
        (y == 0) & (pred_land == 0) & (pred_water == 1)
    )
    reverse = ((y == 1) & (pred_water == 0) & (pred_land == 1)) | (
        (y == 0) & (pred_land == 1) & (pred_water == 0)
    )
    metrics = {
        "delta_waterbird": delta_waterbird,
        "delta_landbird": delta_landbird,
        "background_gap": background_gap,
        "original_flip_rate": float(np.mean(original)),
        "reverse_flip_rate": float(np.mean(reverse)),
    }
    probabilities = {
        "subject_id": subject_ids,
        "label": labels,
        "p_correct_water_background": p_water_background,
        "p_correct_land_background": p_land_background,
        "prediction_water_background": pred_water_background,
        "prediction_land_background": pred_land_background,
    }
    expected = (abs(metrics["delta_waterbird"]) + abs(metrics["delta_landbird"])) / 2.0
    if not np.isclose(metrics["background_gap"], expected, atol=1e-12):
        raise AssertionError("BGGap formula drift detected")
    return metrics, probabilities


def evaluate_model(model: nn.Module, natural_loader: DataLoader, audit_loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = evaluate_accuracy(model, natural_loader, device)
    counterfactual_metrics, probabilities = evaluate_counterfactual(model, audit_loader, device)
    metrics.update(counterfactual_metrics)
    return metrics, probabilities
