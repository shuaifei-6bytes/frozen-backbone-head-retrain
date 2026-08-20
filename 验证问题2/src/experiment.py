"""Waterbirds/FedAvg implementation for Experiment 2 (over-forgetting)."""
from __future__ import annotations

import copy
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

GROUPS = ("WB-Water", "WB-Land", "LB-Land", "LB-Water")
CLIENT_FRACTIONS = (0.50, 0.125, 0.125, 0.125, 0.125)
FULL_A = {"WB-Water": .475, "WB-Land": .025, "LB-Land": .475, "LB-Water": .025}
NEUTRAL = {group: .25 for group in GROUPS}


@dataclass(frozen=True)
class Record:
    image_id: str
    image_path: str
    label: int
    background: int
    split: str

    @property
    def group(self) -> str:
        return ("WB" if self.label else "LB") + "-" + ("Water" if self.background else "Land")


@dataclass(frozen=True)
class Assignment:
    image_id: str
    image_path: str
    label: int
    background: int
    client_id: int
    occurrence: int

    @property
    def group(self) -> str:
        return ("WB" if self.label else "LB") + "-" + ("Water" if self.background else "Land")


def _split(value: Any) -> str:
    return {"0": "train", "1": "val", "2": "test", "train": "train", "val": "val", "test": "test"}.get(str(value), "")


def load_records(data_dir: Path) -> list[Record]:
    metadata = data_dir / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"Waterbirds metadata.csv not found: {metadata}")
    frame = pd.read_csv(metadata)
    required = {"img_id", "img_filename", "y", "place", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metadata.csv lacks columns: {sorted(missing)}")
    records = []
    for row in frame.itertuples(index=False):
        data = row._asdict()
        image = data_dir / str(data["img_filename"]).replace("\\", "/").lstrip("/")
        records.append(Record(str(data["img_id"]), str(image), int(data["y"]), int(data["place"]), _split(data["split"])))
    absent = [r.image_path for r in records if not Path(r.image_path).is_file()]
    if absent:
        raise FileNotFoundError("Missing Waterbirds images, e.g. " + absent[0])
    if any(not r.split for r in records):
        raise ValueError("Unknown metadata split code")
    return records


def _counts(total: int, distribution: Mapping[str, float]) -> dict[str, int]:
    raw = {g: total * distribution[g] for g in GROUPS}
    result = {g: int(math.floor(raw[g])) for g in GROUPS}
    for g in sorted(GROUPS, key=lambda x: (-(raw[x] - result[x]), x))[: total - sum(result.values())]: result[g] += 1
    return result


def _client_sizes(total: int) -> list[int]:
    raw = [total * x for x in CLIENT_FRACTIONS]; sizes = [int(x) for x in raw]
    for i in sorted(range(5), key=lambda i: (-(raw[i] - sizes[i]), i))[: total - sum(sizes)]: sizes[i] += 1
    return sizes


def build_source_pools(records: list[Record], seed: int) -> dict[int, list[Record]]:
    """Disjoint 50%/12.5% pools that cover training records exactly."""
    train = [r for r in records if r.split == "train"]
    sizes, rng = _client_sizes(len(train)), np.random.RandomState(seed)
    by_group = {g: [r for r in train if r.group == g] for g in GROUPS}
    pools = {i: [] for i in range(5)}
    # Proportional allocation, followed by deterministic total-size rebalancing.
    for group, items in by_group.items():
        indices = np.arange(len(items)); rng.shuffle(indices)
        raw = [len(items) * f for f in CLIENT_FRACTIONS]; ns = [int(x) for x in raw]
        for i in sorted(range(5), key=lambda i: (-(raw[i] - ns[i]), i))[:len(items)-sum(ns)]: ns[i] += 1
        offset = 0
        for i, n in enumerate(ns): pools[i].extend(items[j] for j in indices[offset:offset+n]); offset += n
    while [len(pools[i]) for i in range(5)] != sizes:
        receiver = next(i for i in range(5) if len(pools[i]) < sizes[i]); donor = next(i for i in range(5) if len(pools[i]) > sizes[i])
        item = pools[donor].pop(); pools[receiver].append(item)
    if sum(map(len, pools.values())) != len(train): raise AssertionError("source pools do not cover train set")
    return pools


def sample_assignments(pools: Mapping[int, list[Record]], seed: int, a_distribution: Mapping[str, float]) -> dict[int, list[Assignment]]:
    """Fixed-size group resampling. B--E are neutral; A is configurable."""
    out: dict[int, list[Assignment]] = {}
    for client_id, pool in pools.items():
        distribution = a_distribution if client_id == 0 else NEUTRAL
        rng = np.random.RandomState(seed + 1009 * (client_id + 1))
        selected: list[Record] = []
        for group, n in _counts(len(pool), distribution).items():
            candidates = [r for r in pool if r.group == group]
            if not candidates: raise ValueError(f"Client {client_id} has no {group} records")
            choice = rng.choice(len(candidates), n, replace=n > len(candidates))
            selected.extend(candidates[i] for i in choice)
        rng.shuffle(selected)
        out[client_id] = [Assignment(r.image_id, r.image_path, r.label, r.background, client_id, j) for j, r in enumerate(selected)]
    return out


class ImageDataset(Dataset):
    def __init__(self, items: Iterable[Record | Assignment], image_size: int, train: bool = False):
        self.items = list(items)
        ops: list[Any] = [transforms.Resize((image_size, image_size))]
        if train: ops.append(transforms.RandomHorizontalFlip())
        ops += [transforms.ToTensor(), transforms.Normalize((.485,.456,.406), (.229,.224,.225))]
        self.transform = transforms.Compose(ops)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        x = self.items[index]
        return {"image": self.transform(Image.open(x.image_path).convert("RGB")), "label": torch.tensor(x.label), "group": x.group, "image_id": x.image_id}


def loaders(assignments: Mapping[int, list[Assignment]], batch_size: int, image_size: int, workers: int, seed: int) -> list[DataLoader]:
    return [DataLoader(ImageDataset(assignments[client_id], image_size, True), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, generator=torch.Generator().manual_seed(seed + client_id)) for client_id in sorted(assignments)]


def make_model() -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def fedavg(init_state: Mapping[str, torch.Tensor], client_loaders: list[DataLoader], rounds: int, lr: float, device: torch.device) -> tuple[nn.Module, list[dict[str, float]]]:
    global_model = make_model().to(device); global_model.load_state_dict(init_state)
    history = []
    for rnd in range(rounds):
        local_states, weights, losses = [], [], []
        for loader in client_loaders:
            local = copy.deepcopy(global_model); local.train()
            optimizer = torch.optim.SGD((p for p in local.parameters() if p.requires_grad), lr=lr, momentum=.9, weight_decay=1e-4)
            loss_sum = 0.0
            for batch in loader:
                optimizer.zero_grad(); logits = local(batch["image"].to(device)); loss = nn.functional.cross_entropy(logits, batch["label"].to(device)); loss.backward(); optimizer.step()
                loss_sum += loss.item() * len(batch["label"])
            local_states.append({k:v.detach().cpu() for k,v in local.state_dict().items()}); weights.append(len(loader.dataset)); losses.append(loss_sum / len(loader.dataset))
        total = sum(weights); averaged = {k: sum(state[k] * (w / total) for state, w in zip(local_states, weights)) for k in local_states[0]}
        mean_loss = float(np.mean(losses))
        global_model.load_state_dict(averaged); history.append({"round": rnd+1, "mean_client_loss": mean_loss})
        print(f"[FedAvg] round {rnd + 1}/{rounds} | mean client loss={mean_loss:.6f}", flush=True)
    return global_model, history


@torch.no_grad()
def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, device: torch.device) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    correct = total = 0; groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    model.eval()
    for batch in loader:
        pred = model(batch["image"].to(device)).argmax(1).cpu(); y = batch["label"]
        correct += int((pred == y).sum()); total += len(y)
        for p, truth, group in zip(pred.tolist(), y.tolist(), batch["group"]): groups[group][0] += int(p == truth); groups[group][1] += 1
    result = {"overall_accuracy": correct / total, **{f"group_accuracy_{g}": c/n for g,(c,n) in groups.items()}}
    result["worst_group_accuracy"] = min(result[f"group_accuracy_{g}"] for g in groups)
    return result


@torch.no_grad()
def relation_metrics(model: nn.Module, records: list[Record], image_size: int, batch_size: int, device: torch.device) -> dict[str, float]:
    """Approximate same-label, alternate-background pairs from the test split."""
    test = [r for r in records if r.split == "test"]; rng = np.random.RandomState(913)
    per_label = {}
    for label in (0, 1):
        water = [r for r in test if r.label == label and r.background == 1]; land = [r for r in test if r.label == label and r.background == 0]
        n = min(len(water), len(land)); rng.shuffle(water); rng.shuffle(land); per_label[label] = list(zip(water[:n], land[:n]))
    result = {}
    for label, pairs in per_label.items():
        water = evaluate(model, ImageDataset([a for a,_ in pairs], image_size), batch_size, device)["overall_accuracy"]
        land = evaluate(model, ImageDataset([b for _,b in pairs], image_size), batch_size, device)["overall_accuracy"]
        name = "waterbird" if label else "landbird"; result[f"delta_{name}"] = water - land
        # With true label fixed, water->land and land->water prediction changes are directional flips.
        wa = DataLoader(ImageDataset([a for a,_ in pairs], image_size), batch_size=batch_size); la = DataLoader(ImageDataset([b for _,b in pairs], image_size), batch_size=batch_size)
        original = reverse = n_items = 0
        for bw, bl in zip(wa, la):
            pw = model(bw["image"].to(device)).argmax(1).cpu(); pl = model(bl["image"].to(device)).argmax(1).cpu()
            original += int(((pw == label) & (pl != label)).sum()); reverse += int(((pw != label) & (pl == label)).sum()); n_items += len(pw)
        result[f"original_flip_rate_{name}"] = original/n_items; result[f"reverse_flip_rate_{name}"] = reverse/n_items
    result["background_gap"] = (abs(result["delta_waterbird"]) + abs(result["delta_landbird"])) / 2
    result["original_flip_rate"] = (result["original_flip_rate_waterbird"] + result["original_flip_rate_landbird"]) / 2
    result["reverse_flip_rate"] = (result["reverse_flip_rate_waterbird"] + result["reverse_flip_rate_landbird"]) / 2
    return result


def evaluation_sets(records: list[Record], seed: int) -> tuple[ImageDataset, dict[int, ImageDataset]]:
    val = [r for r in records if r.split == "val"]; rng = np.random.RandomState(seed + 8000)
    neutral_a: list[Record] = []
    for group in GROUPS:
        choices = [r for r in val if r.group == group]; neutral_a.extend(choices[i] for i in rng.choice(len(choices), min(len(choices), min(len([x for x in val if x.group == g]) for g in GROUPS)), replace=False))
    by_client = {i: [] for i in range(1, 5)}
    for group in GROUPS:
        items = [r for r in val if r.group == group]; rng.shuffle(items)
        for j, item in enumerate(items): by_client[(j % 4) + 1].append(item)
    return ImageDataset(neutral_a, 160), {i: ImageDataset(x, 160) for i, x in by_client.items()}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_rows(assignments: Mapping[int, list[Assignment]], condition: str) -> list[dict[str, Any]]:
    return [{"condition": condition, **asdict(item), "group": item.group} for group in assignments.values() for item in group]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
