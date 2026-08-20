"""Composite-only Waterbirds sampling for the approximate attribution study.

The Kaggle input contains already-composited Waterbirds images. It does not
contain CUB foregrounds, masks, or Places backgrounds, so this module keeps the
available images intact and changes group sampling frequencies. This is a
distribution-intervention approximation, not strict background replacement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLIENT_A_FULL, CLIENT_DATA_FRACTIONS, CLIENT_NEUTRAL, GROUPS


LABEL_NAME = {1: "WB", 0: "LB"}
BACKGROUND_NAME = {1: "Water", 0: "Land"}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    split: str
    image_path: str
    background: int

    @property
    def group(self) -> str:
        return f"{LABEL_NAME[self.label]}-{BACKGROUND_NAME[self.background]}"


@dataclass(frozen=True)
class Assignment:
    occurrence_id: int
    subject_id: str
    client_id: int
    label: int
    background: int
    image_path: str

    @property
    def group(self) -> str:
        return f"{LABEL_NAME[self.label]}-{BACKGROUND_NAME[self.background]}"


def _split_name(value: Any) -> str:
    if str(value) in {"0", "0.0", "train"}:
        return "train"
    if str(value) in {"1", "1.0", "val"}:
        return "val"
    if str(value) in {"2", "2.0", "test"}:
        return "test"
    raise ValueError(f"Unknown split value: {value!r}")


def load_subject_records(data_dir: Path) -> list[SubjectRecord]:
    """Load the six-column metadata in the existing Kaggle dataset."""
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    required = {"img_id", "img_filename", "y", "split", "place"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    records: list[SubjectRecord] = []
    missing_images: list[str] = []
    for _, row in frame.iterrows():
        relative = str(row["img_filename"]).replace("\\", "/").lstrip("/")
        image_path = data_dir / relative
        if not image_path.is_file():
            missing_images.append(str(image_path))
        records.append(
            SubjectRecord(
                subject_id=str(row["img_id"]),
                label=int(row["y"]),
                split=_split_name(row["split"]),
                image_path=str(image_path),
                background=int(row["place"]),
            )
        )
    if missing_images:
        raise FileNotFoundError("Missing Waterbirds images, first entries:\n" + "\n".join(missing_images[:10]))
    ids = [record.subject_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("img_id must be unique")
    return records


def _client_target_sizes(total: int, num_clients: int) -> dict[int, int]:
    if num_clients != len(CLIENT_DATA_FRACTIONS) or set(CLIENT_DATA_FRACTIONS) != set(range(num_clients)):
        raise ValueError("This locked stress test requires Client A plus Clients B--E.")
    if not np.isclose(sum(CLIENT_DATA_FRACTIONS.values()), 1.0):
        raise ValueError("CLIENT_DATA_FRACTIONS must sum to one.")
    raw = {client_id: total * CLIENT_DATA_FRACTIONS[client_id] for client_id in range(num_clients)}
    counts = {client_id: int(math.floor(raw[client_id])) for client_id in range(num_clients)}
    remainder = total - sum(counts.values())
    for client_id in sorted(
        range(num_clients),
        key=lambda key: (-(raw[key] - counts[key]), key),
    )[:remainder]:
        counts[client_id] += 1
    return counts


def _source_pool_counts(
    train: Sequence[SubjectRecord],
    num_clients: int,
) -> dict[str, dict[int, int]]:
    """Allocate all original groups while meeting exact global client totals."""
    target_totals = _client_target_sizes(len(train), num_clients)
    counts: dict[str, dict[int, int]] = {}
    actual = {client_id: 0 for client_id in range(num_clients)}
    for group in GROUPS:
        group_total = sum(record.group == group for record in train)
        raw = {client_id: group_total * CLIENT_DATA_FRACTIONS[client_id] for client_id in range(num_clients)}
        allocation = {client_id: int(math.floor(raw[client_id])) for client_id in range(num_clients)}
        for client_id in sorted(
            range(num_clients),
            key=lambda key: (-(raw[key] - allocation[key]), key),
        )[: group_total - sum(allocation.values())]:
            allocation[client_id] += 1
        counts[group] = allocation
        for client_id in range(num_clients):
            actual[client_id] += allocation[client_id]

    # Per-group rounding can move a few records away from the exact 50/12.5%
    # totals. Reassign only source-pool ownership until global totals match.
    deficit = {client_id: target_totals[client_id] - actual[client_id] for client_id in range(num_clients)}
    while any(value != 0 for value in deficit.values()):
        receiver = next(client_id for client_id in range(num_clients) if deficit[client_id] > 0)
        donor = next(client_id for client_id in range(num_clients) if deficit[client_id] < 0)
        candidates = [group for group in GROUPS if counts[group][donor] > 0]
        if not candidates:
            raise RuntimeError("Could not rebalance client source-pool allocation.")
        group = max(
            candidates,
            key=lambda name: (
                sum(record.group == name for record in train) * CLIENT_DATA_FRACTIONS[receiver]
                - counts[name][receiver]
            ),
        )
        counts[group][donor] -= 1
        counts[group][receiver] += 1
        deficit[donor] += 1
        deficit[receiver] -= 1
    return counts


def stratified_client_split(records: Sequence[SubjectRecord], num_clients: int, seed: int) -> dict[int, list[str]]:
    """Create the 50% Client-A and 12.5%-each Client-B--E source pools."""
    train = [record for record in records if record.split == "train"]
    rng = np.random.RandomState(seed)
    clients: dict[int, list[str]] = {client_id: [] for client_id in range(num_clients)}
    source_counts = _source_pool_counts(train, num_clients)
    for group in GROUPS:
        ids = np.asarray([record.subject_id for record in train if record.group == group], dtype=object)
        rng.shuffle(ids)
        offset = 0
        for client_id in range(num_clients):
            count = source_counts[group][client_id]
            clients[client_id].extend(str(value) for value in ids[offset : offset + count].tolist())
            offset += count
        if offset != len(ids):
            raise AssertionError(f"Source-pool allocation did not cover all {group} records")
    records_by_id = {record.subject_id: record for record in train}
    for client_id in clients:
        clients[client_id].sort(key=int)
        groups_present = {records_by_id[sid].group for sid in clients[client_id]}
        if groups_present != set(GROUPS):
            raise ValueError(f"Client {client_id} source pool does not contain all four groups")
    covered = set().union(*(set(values) for values in clients.values()))
    if covered != set(records_by_id):
        raise AssertionError("Client split does not exactly cover the training images")
    actual_sizes = {client_id: len(client_ids) for client_id, client_ids in clients.items()}
    expected_sizes = _client_target_sizes(len(train), num_clients)
    if actual_sizes != expected_sizes:
        raise AssertionError(f"Client source-pool sizes differ from design: {actual_sizes} != {expected_sizes}")
    return clients


def _largest_remainder(total: int, distribution: Mapping[str, float]) -> dict[str, int]:
    raw = {group: total * float(distribution[group]) for group in GROUPS}
    counts = {group: int(math.floor(raw[group])) for group in GROUPS}
    remainder = total - sum(counts.values())
    order = sorted(GROUPS, key=lambda group: (-(raw[group] - counts[group]), group))
    for group in order[:remainder]:
        counts[group] += 1
    return counts


def _resample_client(
    source_ids: Sequence[str],
    records_by_id: Mapping[str, SubjectRecord],
    distribution: Mapping[str, float],
    client_id: int,
    seed: int,
) -> list[Assignment]:
    """Create a fixed epoch manifest, sampling with replacement where needed."""
    rng = np.random.RandomState(seed)
    pools = {
        group: np.asarray([sid for sid in source_ids if records_by_id[sid].group == group], dtype=object)
        for group in GROUPS
    }
    if any(len(pool) == 0 for pool in pools.values()):
        raise ValueError(f"Client {client_id} has an empty group pool")
    counts = _largest_remainder(len(source_ids), distribution)
    selected: list[str] = []
    for group in GROUPS:
        chosen = rng.choice(pools[group], size=counts[group], replace=counts[group] > len(pools[group]))
        selected.extend(str(value) for value in chosen.tolist())
    rng.shuffle(selected)
    assignments: list[Assignment] = []
    for occurrence_id, subject_id in enumerate(selected):
        record = records_by_id[subject_id]
        assignments.append(
            Assignment(
                occurrence_id=occurrence_id,
                subject_id=subject_id,
                client_id=client_id,
                label=record.label,
                background=record.background,
                image_path=record.image_path,
            )
        )
    return assignments


def build_paired_assignments(
    records: Sequence[SubjectRecord], client_split: Mapping[int, Sequence[str]], seed: int
) -> tuple[dict[int, list[Assignment]], dict[int, list[Assignment]]]:
    """Build both interventions; non-target clients are byte-identical."""
    records_by_id = {record.subject_id: record for record in records}
    full: dict[int, list[Assignment]] = {}
    neutral: dict[int, list[Assignment]] = {}
    for client_id, source_ids in client_split.items():
        if client_id == 0:
            full[client_id] = _resample_client(source_ids, records_by_id, CLIENT_A_FULL, client_id, seed + 1000)
            neutral[client_id] = _resample_client(source_ids, records_by_id, CLIENT_NEUTRAL, client_id, seed + 2000)
        else:
            shared = _resample_client(source_ids, records_by_id, CLIENT_NEUTRAL, client_id, seed + 3000 + client_id)
            full[client_id] = shared
            neutral[client_id] = list(shared)
    for client_id, source_ids in client_split.items():
        if len(full[client_id]) != len(source_ids) or len(neutral[client_id]) != len(source_ids):
            raise AssertionError(f"Client {client_id} sample count changed")
        allowed = set(source_ids)
        if any(item.subject_id not in allowed for item in full[client_id] + neutral[client_id]):
            raise AssertionError(f"Client {client_id} sampled outside its fixed source pool")
        if client_id != 0 and full[client_id] != neutral[client_id]:
            raise AssertionError(f"Non-target client {client_id} changed")
    return full, neutral


def assignment_description(
    split: Mapping[int, Sequence[str]],
    full: Mapping[int, Sequence[Assignment]],
    neutral: Mapping[int, Sequence[Assignment]],
    records: Sequence[SubjectRecord],
) -> dict[str, Any]:
    """Produce machine-readable allocation and evidence-scope records."""
    records_by_id = {record.subject_id: record for record in records}

    def group_counts(items: Sequence[SubjectRecord] | Sequence[Assignment]) -> dict[str, int]:
        return {group: sum(item.group == group for item in items) for group in GROUPS}

    total = sum(len(source_ids) for source_ids in split.values())
    source_pools = {
        str(client_id): {
            "n": len(source_ids),
            "fraction_of_all_train_records": len(source_ids) / total,
            "target_fraction": CLIENT_DATA_FRACTIONS[client_id],
            "group_counts": group_counts([records_by_id[subject_id] for subject_id in source_ids]),
        }
        for client_id, source_ids in split.items()
    }
    assignments = {
        condition: {
            str(client_id): {
                "n": len(items),
                "group_counts": group_counts(items),
                "group_fractions": {
                    group: group_counts(items)[group] / len(items) for group in GROUPS
                },
            }
            for client_id, items in by_client.items()
        }
        for condition, by_client in {"M_full": full, "M_A_neutral": neutral}.items()
    }
    full_a_ids = {item.subject_id for item in full[0]}
    neutral_a_ids = {item.subject_id for item in neutral[0]}
    return {
        "source_pool_allocation": source_pools,
        "training_relation_assignments": assignments,
        "paired_control_status": {
            "same_client_a_sample_count": len(full[0]) == len(neutral[0]),
            "same_client_a_label_count": {
                "LB": sum(item.label == 0 for item in full[0]) == sum(item.label == 0 for item in neutral[0]),
                "WB": sum(item.label == 1 for item in full[0]) == sum(item.label == 1 for item in neutral[0]),
            },
            "same_client_a_image_ids": full_a_ids == neutral_a_ids,
            "strict_same_subject_background_swap_available": False,
            "design_compliance": "approximate_distribution_intervention_only",
            "limitation": (
                "Composite Waterbirds metadata has no foreground-subject identifier or alternative "
                "background rendering. The two Client-A manifests keep total sample count and "
                "label count, but cannot retain identical image IDs while changing background relation."
            ),
        },
    }


def image_transform(image_size: int, train: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)


class AssignedCompositeDataset(Dataset):
    def __init__(self, assignments: Sequence[Assignment], image_size: int, train: bool) -> None:
        self.assignments = list(assignments)
        self.transform = image_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.assignments[index]
        return {
            "image": self.transform(Image.open(item.image_path).convert("RGB")),
            "label": torch.tensor(item.label, dtype=torch.long),
            "group": item.group,
            "subject_id": item.subject_id,
        }


class NaturalTestDataset(Dataset):
    def __init__(self, records: Sequence[SubjectRecord], image_size: int) -> None:
        self.records = [record for record in records if record.split == "test"]
        self.transform = image_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "image": self.transform(Image.open(record.image_path).convert("RGB")),
            "label": torch.tensor(record.label, dtype=torch.long),
            "group": record.group,
            "subject_id": record.subject_id,
        }


class ApproximateCounterfactualDataset(Dataset):
    """Fixed same-class/different-image background pairs from test data."""

    def __init__(self, records: Sequence[SubjectRecord], image_size: int, pairs_per_class: int, seed: int) -> None:
        test = [record for record in records if record.split == "test"]
        rng = np.random.RandomState(seed)
        self.items: list[tuple[SubjectRecord, SubjectRecord]] = []
        for label in (0, 1):
            water = [record for record in test if record.label == label and record.background == 1]
            land = [record for record in test if record.label == label and record.background == 0]
            count = min(pairs_per_class, len(water), len(land))
            water_indices = rng.choice(len(water), size=count, replace=False)
            land_indices = rng.choice(len(land), size=count, replace=False)
            self.items.extend((water[i], land[j]) for i, j in zip(water_indices, land_indices))
        self.items.sort(key=lambda pair: (pair[0].label, int(pair[0].subject_id), int(pair[1].subject_id)))
        self.transform = image_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        water, land = self.items[index]
        return {
            "water": self.transform(Image.open(water.image_path).convert("RGB")),
            "land": self.transform(Image.open(land.image_path).convert("RGB")),
            "label": torch.tensor(water.label, dtype=torch.long),
            "subject_id": f"{water.subject_id}:{land.subject_id}",
        }

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "label": water.label,
                "water_image_id": water.subject_id,
                "water_image_path": water.image_path,
                "land_image_id": land.subject_id,
                "land_image_path": land.image_path,
                "same_subject": False,
            }
            for water, land in self.items
        ]


def make_client_loaders(
    records: Sequence[SubjectRecord],
    assignments: Mapping[int, Sequence[Assignment]],
    batch_size: int,
    image_size: int,
    num_workers: int,
    seed: int,
) -> list[DataLoader]:
    del records
    loaders: list[DataLoader] = []
    for client_id in sorted(assignments):
        generator = torch.Generator().manual_seed(seed + client_id)
        dataset = AssignedCompositeDataset(assignments[client_id], image_size, train=True)
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=True,
        ))
    return loaders


def assignments_to_rows(assignments: Mapping[int, Sequence[Assignment]], condition: str) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "client_id": item.client_id,
            "occurrence_id": item.occurrence_id,
            "source_image_id": item.subject_id,
            "label": item.label,
            "background": item.background,
            "group": item.group,
            "image_path": item.image_path,
        }
        for client_id in sorted(assignments)
        for item in assignments[client_id]
    ]
