"""Strict subject-preserving Waterbirds composition and paired manifests.

The standard released Waterbirds directory contains already-composited images.
That is sufficient for ordinary classification, but not for this experiment:
neutralising a relation while retaining exactly the same subjects requires the
original foreground image, its segmentation mask, and the background image.
This module therefore rejects composite-only input instead of silently treating
different birds as a counterfactual pair.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import CLIENT_A_FULL, CLIENT_NEUTRAL, GROUPS


LABEL_NAME = {1: "WB", 0: "LB"}
BACKGROUND_NAME = {1: "Water", 0: "Land"}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    split: str
    subject_path: str
    mask_path: str
    original_background_path: str
    original_composite_path: str | None
    original_background: int


@dataclass(frozen=True)
class Assignment:
    subject_id: str
    client_id: int
    label: int
    background: int
    background_path: str

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


def _resolve(root: Path, value: Any) -> Path:
    raw = str(value).replace("\\", "/").lstrip("/")
    path = Path(value)
    # Waterbirds place_filename values conventionally start with "/" even
    # though they are relative Places paths (for example /o/ocean/x.jpg).
    # Treat a path as genuinely absolute only when that file exists.
    return path if path.is_absolute() and path.is_file() else root / raw


def _first_existing(row: pd.Series, columns: Sequence[str]) -> Any | None:
    for column in columns:
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return row[column]
    return None


def load_subject_records(data_dir: Path) -> list[SubjectRecord]:
    """Load the canonical metadata and validate strict-composition assets.

    Supported metadata columns (aliases accepted):
      subject_id/img_id, y/waterbird, split,
      subject_filename/foreground_filename,
      mask_filename/segmentation_filename,
      place_filename/background_filename, place/water_background,
      optional img_filename/original_composite_filename.

    Relative subject, mask and background paths are resolved against
    ``data_dir/subjects``, ``data_dir/masks`` and ``data_dir/backgrounds``.
    """
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    records: list[SubjectRecord] = []
    missing: list[str] = []
    for index, row in frame.iterrows():
        subject_id = str(_first_existing(row, ("subject_id", "img_id")) or index)
        label_raw = _first_existing(row, ("waterbird", "y", "label"))
        bg_raw = _first_existing(row, ("water_background", "place", "background"))
        subject_rel = _first_existing(row, ("subject_filename", "foreground_filename"))
        mask_rel = _first_existing(row, ("mask_filename", "segmentation_filename"))
        background_rel = _first_existing(row, ("background_filename", "place_filename"))
        composite_rel = _first_existing(row, ("original_composite_filename", "img_filename", "img_path"))
        if label_raw is None or bg_raw is None or subject_rel is None or mask_rel is None or background_rel is None:
            missing.append(subject_id)
            continue
        subject_path = _resolve(data_dir / "subjects", subject_rel)
        mask_path = _resolve(data_dir / "masks", mask_rel)
        background_path = _resolve(data_dir / "backgrounds", background_rel)
        composite_path = _resolve(data_dir, composite_rel) if composite_rel is not None else None
        for kind, path in (("subject", subject_path), ("mask", mask_path), ("background", background_path)):
            if not path.is_file():
                missing.append(f"{subject_id}:{kind}:{path}")
        records.append(
            SubjectRecord(
                subject_id=subject_id,
                label=int(label_raw),
                split=_split_name(row.get("split", "train")),
                subject_path=str(subject_path),
                mask_path=str(mask_path),
                original_background_path=str(background_path),
                original_composite_path=str(composite_path) if composite_path and composite_path.is_file() else None,
                original_background=int(bg_raw),
            )
        )
    if missing:
        preview = "\n  ".join(missing[:12])
        raise ValueError(
            "Strict same-subject background reassignment cannot be built from the supplied data. "
            "Provide subject_filename, mask_filename and place_filename/background_filename assets. "
            f"First unresolved rows/assets:\n  {preview}"
        )
    if not records:
        raise ValueError("metadata.csv contains no usable records")
    ids = [record.subject_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("subject_id/img_id must be unique")
    return records


def stratified_client_split(records: Sequence[SubjectRecord], num_clients: int, seed: int) -> dict[int, list[str]]:
    """Make one fixed label-stratified split shared by both trajectories."""
    train = [record for record in records if record.split == "train"]
    rng = np.random.RandomState(seed)
    clients: dict[int, list[str]] = {client_id: [] for client_id in range(num_clients)}
    for label in (0, 1):
        ids = np.array([record.subject_id for record in train if record.label == label], dtype=object)
        rng.shuffle(ids)
        for client_id, part in enumerate(np.array_split(ids, num_clients)):
            clients[client_id].extend(str(value) for value in part.tolist())
    for client_id in clients:
        clients[client_id].sort()
    if set().union(*(set(values) for values in clients.values())) != {record.subject_id for record in train}:
        raise AssertionError("Client split does not exactly cover the training subjects")
    return clients


def _largest_remainder(total: int, probabilities: Mapping[int, float]) -> dict[int, int]:
    raw = {key: total * probability for key, probability in probabilities.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(probabilities, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _background_pools(records: Sequence[SubjectRecord]) -> dict[int, list[str]]:
    pools = {0: [], 1: []}
    for record in records:
        if record.split == "train":
            pools[record.original_background].append(record.original_background_path)
    if not all(pools.values()):
        raise ValueError("Training metadata must provide at least one Water and one Land background")
    return pools


def _assign_client(
    subject_ids: Sequence[str],
    records_by_id: Mapping[str, SubjectRecord],
    distribution: Mapping[str, float],
    pools: Mapping[int, Sequence[str]],
    client_id: int,
    rng: np.random.RandomState,
) -> list[Assignment]:
    assignments: list[Assignment] = []
    for label in (0, 1):
        ids = np.array([sid for sid in subject_ids if records_by_id[sid].label == label], dtype=object)
        rng.shuffle(ids)
        group_probabilities = {
            background: distribution[f"{LABEL_NAME[label]}-{BACKGROUND_NAME[background]}"] / 0.5
            for background in (0, 1)
        }
        counts = _largest_remainder(len(ids), group_probabilities)
        cursor = 0
        for background in (0, 1):
            chosen = ids[cursor : cursor + counts[background]]
            cursor += counts[background]
            background_pool = np.asarray(pools[background], dtype=object)
            sampled = rng.choice(background_pool, size=len(chosen), replace=len(chosen) > len(background_pool))
            assignments.extend(
                Assignment(str(sid), client_id, label, background, str(bg_path))
                for sid, bg_path in zip(chosen, sampled)
            )
    assignments.sort(key=lambda item: item.subject_id)
    return assignments


def build_paired_assignments(
    records: Sequence[SubjectRecord], client_split: Mapping[int, Sequence[str]], seed: int
) -> tuple[dict[int, list[Assignment]], dict[int, list[Assignment]]]:
    """Build full and neutral assignments; clients 1--4 are byte-identical."""
    records_by_id = {record.subject_id: record for record in records}
    pools = _background_pools(records)
    full: dict[int, list[Assignment]] = {}
    neutral: dict[int, list[Assignment]] = {}
    for client_id, subject_ids in client_split.items():
        if client_id == 0:
            full[client_id] = _assign_client(
                subject_ids, records_by_id, CLIENT_A_FULL, pools, client_id, np.random.RandomState(seed + 1000)
            )
            neutral[client_id] = _assign_client(
                subject_ids, records_by_id, CLIENT_NEUTRAL, pools, client_id, np.random.RandomState(seed + 2000)
            )
        else:
            shared = _assign_client(
                subject_ids,
                records_by_id,
                CLIENT_NEUTRAL,
                pools,
                client_id,
                np.random.RandomState(seed + 3000 + client_id),
            )
            full[client_id] = shared
            neutral[client_id] = list(shared)
    _validate_paired_assignments(client_split, full, neutral)
    return full, neutral


def _validate_paired_assignments(
    split: Mapping[int, Sequence[str]],
    full: Mapping[int, Sequence[Assignment]],
    neutral: Mapping[int, Sequence[Assignment]],
) -> None:
    for client_id, expected in split.items():
        full_ids = [item.subject_id for item in full[client_id]]
        neutral_ids = [item.subject_id for item in neutral[client_id]]
        if sorted(expected) != sorted(full_ids) or sorted(expected) != sorted(neutral_ids):
            raise AssertionError(f"Subject mismatch for client {client_id}")
        if client_id != 0 and full[client_id] != neutral[client_id]:
            raise AssertionError(f"Non-target client {client_id} changed between trajectories")


def compose(subject_path: str, mask_path: str, background_path: str) -> Image.Image:
    subject = Image.open(subject_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    background = Image.open(background_path).convert("RGB").resize(subject.size, Image.Resampling.BILINEAR)
    if mask.size != subject.size:
        mask = mask.resize(subject.size, Image.Resampling.NEAREST)
    return Image.composite(subject, background, mask)


def image_transform(image_size: int, train: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((image_size, image_size))]
    if train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


class AssignedCompositeDataset(Dataset):
    def __init__(
        self,
        records_by_id: Mapping[str, SubjectRecord],
        assignments: Sequence[Assignment],
        image_size: int,
        train: bool,
    ) -> None:
        self.records = records_by_id
        self.assignments = list(assignments)
        self.transform = image_transform(image_size, train=train)

    def __len__(self) -> int:
        return len(self.assignments)

    def __getitem__(self, index: int) -> dict[str, Any]:
        assignment = self.assignments[index]
        record = self.records[assignment.subject_id]
        image = compose(record.subject_path, record.mask_path, assignment.background_path)
        return {
            "image": self.transform(image),
            "label": torch.tensor(record.label, dtype=torch.long),
            "group": assignment.group,
            "subject_id": record.subject_id,
        }


class NaturalTestDataset(Dataset):
    def __init__(self, records: Sequence[SubjectRecord], image_size: int) -> None:
        self.records = [record for record in records if record.split == "test"]
        self.transform = image_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record.original_composite_path:
            image = Image.open(record.original_composite_path).convert("RGB")
        else:
            image = compose(record.subject_path, record.mask_path, record.original_background_path)
        group = f"{LABEL_NAME[record.label]}-{BACKGROUND_NAME[record.original_background]}"
        return {
            "image": self.transform(image),
            "label": torch.tensor(record.label, dtype=torch.long),
            "group": group,
            "subject_id": record.subject_id,
        }


class StrictCounterfactualDataset(Dataset):
    """Every item is one subject rendered on both background types."""

    def __init__(
        self,
        records: Sequence[SubjectRecord],
        image_size: int,
        subjects_per_class: int,
        seed: int,
    ) -> None:
        test = [record for record in records if record.split == "test"]
        pools = {0: [], 1: []}
        for record in test:
            pools[record.original_background].append(record.original_background_path)
        if not all(pools.values()):
            raise ValueError("Test metadata must provide both Water and Land background pools")
        rng = np.random.RandomState(seed)
        chosen: list[SubjectRecord] = []
        for label in (0, 1):
            candidates = [record for record in test if record.label == label]
            count = min(subjects_per_class, len(candidates))
            indices = rng.choice(len(candidates), size=count, replace=False)
            chosen.extend(candidates[index] for index in indices)
        self.items: list[tuple[SubjectRecord, str, str]] = []
        for record in sorted(chosen, key=lambda item: item.subject_id):
            land = str(rng.choice(np.asarray(pools[0], dtype=object)))
            water = str(rng.choice(np.asarray(pools[1], dtype=object)))
            self.items.append((record, water, land))
        self.transform = image_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record, water_path, land_path = self.items[index]
        return {
            "water": self.transform(compose(record.subject_path, record.mask_path, water_path)),
            "land": self.transform(compose(record.subject_path, record.mask_path, land_path)),
            "label": torch.tensor(record.label, dtype=torch.long),
            "subject_id": record.subject_id,
        }

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "subject_id": record.subject_id,
                "label": record.label,
                "water_background_path": water,
                "land_background_path": land,
            }
            for record, water, land in self.items
        ]


def make_client_loaders(
    records: Sequence[SubjectRecord],
    assignments: Mapping[int, Sequence[Assignment]],
    batch_size: int,
    image_size: int,
    num_workers: int,
    seed: int,
) -> list[DataLoader]:
    records_by_id = {record.subject_id: record for record in records}
    loaders: list[DataLoader] = []
    for client_id in sorted(assignments):
        generator = torch.Generator().manual_seed(seed + client_id)
        dataset = AssignedCompositeDataset(records_by_id, assignments[client_id], image_size, train=True)
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=num_workers,
                pin_memory=True,
            )
        )
    return loaders


def assignments_to_rows(assignments: Mapping[int, Sequence[Assignment]], condition: str) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition,
            "client_id": item.client_id,
            "subject_id": item.subject_id,
            "label": item.label,
            "background": item.background,
            "group": item.group,
            "background_path": item.background_path,
        }
        for client_id in sorted(assignments)
        for item in assignments[client_id]
    ]
