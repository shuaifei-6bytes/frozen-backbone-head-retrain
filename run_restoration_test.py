"""Experiment 3: Relation Restoration Test.

The program deliberately keeps M_global and M_B read-only: only a newly
created classifier head is optimized during restoration.  It writes all
artifacts required by ``实验设计单.md`` and never contains machine-specific paths.

Expected data layout::

    data_dir/
      metadata.csv                  # Waterbirds metadata (img_filename,y,place,split)
      counterfactual_pairs.csv      # subject_id,label,water_image,land_image
      <images named by the CSV files>

``counterfactual_pairs.csv`` is intentionally explicit.  Creating a genuine
same-subject/different-background pair is a data-generation task and must not
be approximated by randomly pairing different birds during evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 128
CLASSIFIER_NAMES = ("fc", "classifier", "head")


class WaterbirdsResNet50(nn.Module):
    """Checkpoint-compatible model used by ``waterbirds_bc_v2`` in this repo."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes)
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.backbone(images))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-global", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--checkpoint-oracle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path,
                        help="Defaults to DATA_DIR/counterfactual_pairs.csv")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--architecture", default="waterbirds_resnet50",
                        choices=("waterbirds_resnet50", "resnet18", "resnet50", "mobilenet_v3_small"),
                        help="Used only when checkpoints are state_dict files.")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--train-oracle-if-missing", action="store_true")
    parser.add_argument("--oracle-epochs", type=int, default=5)
    parser.add_argument("--oracle-lr", type=float, default=1e-3)
    args = parser.parse_args()
    if args.epochs < 0 or args.oracle_epochs < 1 or args.batch_size < 1:
        parser.error("epochs, oracle-epochs and batch-size must be positive (epochs may be zero).")
    return args


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("restoration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(output_dir / "run.log", encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def select_device(requested: str, logger: logging.Logger) -> torch.device:
    device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("torch=%s cuda=%s device=%s", torch.__version__, torch.version.cuda, device)
    if device.type == "cuda":
        logger.info("gpu=%s", torch.cuda.get_device_name(device))
    elif requested == "cuda":
        logger.warning("CUDA was requested but is unavailable; falling back to CPU.")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "waterbirds_resnet50":
        return WaterbirdsResNet50(num_classes=num_classes)
    if architecture == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if architecture == "resnet50":
        return models.resnet50(weights=None, num_classes=num_classes)
    if architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported architecture: {architecture}")


def unwrap_state_dict(payload: Any) -> Mapping[str, Tensor] | None:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping) and all(isinstance(k, str) for k in candidate):
                return candidate
        if payload and all(isinstance(k, str) for k in payload):
            return payload
    return None


def load_checkpoint(path: Path, architecture: str, num_classes: int, device: torch.device) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, nn.Module):
        model = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("model"), nn.Module):
        model = payload["model"]
    else:
        state_dict = unwrap_state_dict(payload)
        if state_dict is None:
            raise TypeError(f"Unsupported checkpoint format in {path}. Use an nn.Module or a state_dict.")
        model = build_model(architecture, num_classes)
        cleaned = {key.removeprefix("module."): value for key, value in state_dict.items()}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch for {path}. Missing={missing}; unexpected={unexpected}. "
                "Pass the architecture that was used to train the checkpoint."
            )
    return model.to(device)


def classifier_accessor(model: nn.Module) -> tuple[nn.Module, str]:
    """Return the module that owns the final classifier and its attribute name."""
    for name in CLASSIFIER_NAMES:
        value = getattr(model, name, None)
        if isinstance(value, nn.Linear):
            return model, name
        if isinstance(value, nn.Sequential) and isinstance(value[-1], nn.Linear):
            return value, str(len(value) - 1)
    raise ValueError("Unable to locate classifier. Expected .fc, .head, or final .classifier Linear layer.")


def replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    owner, name = classifier_accessor(model)
    old = owner[int(name)] if name.isdigit() else getattr(owner, name)
    new = nn.Linear(old.in_features, num_classes, bias=old.bias is not None)
    if name.isdigit():
        owner[int(name)] = new
    else:
        setattr(owner, name, new)
    return new


def freeze_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    owner, name = classifier_accessor(model)
    head = owner[int(name)] if name.isdigit() else getattr(owner, name)
    for parameter in head.parameters():
        parameter.requires_grad_(True)


def backbone_parameter_difference(first: nn.Module, second: nn.Module) -> dict[str, Any]:
    first_owner, first_name = classifier_accessor(first)
    second_owner, second_name = classifier_accessor(second)
    first_head = first_owner[int(first_name)] if first_name.isdigit() else getattr(first_owner, first_name)
    second_head = second_owner[int(second_name)] if second_name.isdigit() else getattr(second_owner, second_name)
    first_head_ids = {id(parameter) for parameter in first_head.parameters()}
    second_head_ids = {id(parameter) for parameter in second_head.parameters()}
    first_parameters = dict(first.named_parameters())
    second_parameters = dict(second.named_parameters())
    diffs: list[float] = []
    compared = 0
    for name, value in first_parameters.items():
        if id(value) in first_head_ids:
            continue
        if name not in second_parameters or id(second_parameters[name]) in second_head_ids:
            raise ValueError(f"Backbone parameter mismatch: {name}")
        other = second_parameters[name]
        if value.shape != other.shape:
            raise ValueError(f"Backbone parameter shape mismatch: {name}")
        diffs.append(float((value.detach().cpu() - other.detach().cpu()).abs().max()))
        compared += 1
    if compared == 0:
        raise RuntimeError("No backbone parameters were compared; classifier detection is incompatible.")
    maximum = max(diffs)
    return {"max_parameter_difference": maximum, "all_parameters_equal": maximum == 0.0,
            "parameters_compared": compared}


@dataclass(frozen=True)
class Record:
    image: Path
    label: int
    place: int


class ImageRecords(Dataset[tuple[Tensor, int]]):
    def __init__(self, records: list[Record], transform: Callable[[Image.Image], Tensor]):
        self.records, self.transform = records, transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        record = self.records[index]
        with Image.open(record.image) as image:
            return self.transform(image.convert("RGB")), record.label


class PairedAudit(Dataset[tuple[Tensor, Tensor, int]]):
    REQUIRED_COLUMNS = {"label", "water_image", "land_image"}

    def __init__(self, manifest: Path, data_dir: Path, transform: Callable[[Image.Image], Tensor]):
        frame = pd.read_csv(manifest)
        missing = self.REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"Audit manifest is missing columns: {sorted(missing)}")
        self.rows, self.data_dir, self.transform = frame.to_dict("records"), data_dir, transform
        if not self.rows:
            raise ValueError("Audit manifest has no paired examples.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        row = self.rows[index]
        paths = (self.data_dir / str(row["water_image"]), self.data_dir / str(row["land_image"]))
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing paired audit image(s): {paths}")
        images: list[Tensor] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(self.transform(image.convert("RGB")))
        return images[0], images[1], int(row["label"])


def image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def read_metadata(data_dir: Path) -> pd.DataFrame:
    metadata_path = data_dir / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata: {metadata_path}")
    frame = pd.read_csv(metadata_path)
    # Match the two metadata variants supported by waterbirds_bc_v2.
    frame = frame.rename(columns={
        "img_path": "img_filename", "waterbird": "y", "water_background": "place",
    })
    if "split" in frame and frame["split"].dtype == object:
        split_codes = {"train": 0, "val": 1, "test": 2}
        frame["split"] = frame["split"].map(split_codes)
    required = {"img_filename", "y", "place", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
    return frame


def make_balanced_records(frame: pd.DataFrame, data_dir: Path, split: int = 0) -> list[Record]:
    """Return a deterministic four-group 25/25/25/25 sample of equal size."""
    subset = frame[frame["split"] == split].copy()
    groups = {(int(y), int(place)): group for (y, place), group in subset.groupby(["y", "place"], sort=True)}
    if set(groups) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("Training split must contain all four (label, place) groups.")
    per_group = min(len(group) for group in groups.values())
    if per_group == 0:
        raise ValueError("A balanced oracle set cannot be empty.")
    records: list[Record] = []
    for key in sorted(groups):
        group = groups[key].sort_values("img_filename").iloc[:per_group]
        records.extend(Record(data_dir / row.img_filename, int(row.y), int(row.place)) for row in group.itertuples())
    return records


def make_shortcut_records(frame: pd.DataFrame, data_dir: Path, seed: int, split: int = 0) -> list[Record]:
    """Create an exactly 47.5/2.5/47.5/2.5 four-group restoration set."""
    subset = frame[frame["split"] == split].copy()
    groups = {(int(y), int(place)): group for (y, place), group in subset.groupby(["y", "place"], sort=True)}
    target_order = ((1, 1), (1, 0), (0, 0), (0, 1))  # WB+water, WB+land, LB+land, LB+water
    if set(groups) != set(target_order):
        raise ValueError("Training split must contain all four (label, place) groups.")
    # A 95/5 ratio needs 19 aligned and 1 conflicting unit.  Use the largest
    # repeatable size allowed by every group, then shuffle once with the seed.
    units = min(len(groups[(1, 1)]) // 19, len(groups[(0, 0)]) // 19,
                len(groups[(1, 0)]), len(groups[(0, 1)]))
    if units < 1:
        raise ValueError("Insufficient samples to form a 95/5 shortcut restoration set.")
    counts = {(1, 1): 19 * units, (1, 0): units, (0, 0): 19 * units, (0, 1): units}
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    for key in target_order:
        rows = groups[key].sort_values("img_filename").iloc[rng.permutation(len(groups[key]))[:counts[key]]]
        records.extend(Record(data_dir / row.img_filename, int(row.y), int(row.place)) for row in rows.itertuples())
    rng.shuffle(records)
    return records


def make_loader(dataset: Dataset[Any], batch_size: int, workers: int, seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), generator=generator, persistent_workers=workers > 0)


@torch.inference_mode()
def evaluate_pairs(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> dict[str, float]:
    model.eval()
    water_probs: list[Tensor] = []
    land_probs: list[Tensor] = []
    labels: list[Tensor] = []
    for water, land, label in loader:
        water_logits, land_logits = model(water.to(device)), model(land.to(device))
        water_probs.append(water_logits.softmax(1).cpu())
        land_probs.append(land_logits.softmax(1).cpu())
        labels.append(label.cpu())
    water_prob, land_prob, y = torch.cat(water_probs), torch.cat(land_probs), torch.cat(labels)
    if set(y.tolist()) - {0, 1}:
        raise ValueError("This experiment expects binary labels: landbird=0, waterbird=1.")
    wb, lb = y == 1, y == 0
    if not bool(wb.any()) or not bool(lb.any()):
        raise ValueError("Paired audit set must contain both waterbird and landbird subjects.")
    delta_wb = (water_prob[wb, 1] - land_prob[wb, 1]).mean().item()
    delta_lb = (land_prob[lb, 0] - water_prob[lb, 0]).mean().item()
    water_pred, land_pred = water_prob.argmax(1), land_prob.argmax(1)
    correct = torch.cat((water_pred == y, land_pred == y)).float().mean().item()
    group_acc = [
        (water_pred[wb] == 1).float().mean().item(), (land_pred[wb] == 1).float().mean().item(),
        (water_pred[lb] == 0).float().mean().item(), (land_pred[lb] == 0).float().mean().item(),
    ]
    return {
        "delta_waterbird": delta_wb, "delta_landbird": delta_lb,
        "background_gap": (abs(delta_wb) + abs(delta_lb)) / 2,
        "original_flip_rate": (water_pred[wb] != land_pred[wb]).float().mean().item(),
        "reverse_flip_rate": (water_pred[lb] != land_pred[lb]).float().mean().item(),
        "overall_accuracy": correct, "worst_group_accuracy": min(group_acc),
    }


def train_head_epoch(model: nn.Module, loader: DataLoader[Any], optimizer: SGD, device: torch.device,
                     head_only: bool = False) -> None:
    # ``model.train()`` would update frozen BatchNorm running statistics.  During
    # restoration the complete backbone, including such buffers, stays frozen.
    model.train()
    if head_only:
        model.eval()
        owner, name = classifier_accessor(model)
        (owner[int(name)] if name.isdigit() else getattr(owner, name)).train()
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images.to(device)), labels.to(device))
        loss.backward()
        optimizer.step()


def train_oracle(args: argparse.Namespace, records: list[Record], device: torch.device) -> nn.Module:
    model = build_model(args.architecture, args.num_classes).to(device)
    loader = make_loader(ImageRecords(records, image_transform()), args.batch_size, args.num_workers, seed=0)
    optimizer = SGD(model.parameters(), lr=args.oracle_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for _ in range(args.oracle_epochs):
        train_head_epoch(model, loader, optimizer, device)
    return model


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_one_seed(args: argparse.Namespace, device: torch.device, logger: logging.Logger,
                     global_model: nn.Module, b_model: nn.Module, oracle_model: nn.Module,
                     frame: pd.DataFrame, audit: PairedAudit, seed: int) -> dict[str, Any]:
    seed_everything(seed)
    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    check = backbone_parameter_difference(global_model, b_model)
    (seed_dir / "backbone_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    logger.info("seed=%d backbone equality: %s", seed, check)
    if not check["all_parameters_equal"]:
        logger.warning("M_global and M_B backbones differ; their curves cannot validate head-only suppression.")

    # New models are deep-copied so a seed never mutates the loaded checkpoints.
    models_by_name = {"M_global": copy.deepcopy(global_model), "M_B": copy.deepcopy(b_model),
                      "M_oracle": copy.deepcopy(oracle_model)}
    seed_everything(seed + 10_000)
    reference = replace_classifier(models_by_name["M_global"], args.num_classes)
    initial_head = copy.deepcopy(reference.state_dict())
    torch.save(initial_head, seed_dir / "head_init.pt")
    for name, model in models_by_name.items():
        if name != "M_global":
            replace_classifier(model, args.num_classes).load_state_dict(initial_head)
        freeze_backbone(model)
        model.to(device)

    restore_records = make_shortcut_records(frame, args.data_dir, seed)
    audit_loader = DataLoader(audit, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    optimizers = {name: SGD((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay)
                  for name, model in models_by_name.items()}
    rows: list[dict[str, Any]] = []
    for epoch in range(args.epochs + 1):
        for name, model in models_by_name.items():
            metrics = evaluate_pairs(model, audit_loader, device)
            rows.append({"epoch": epoch, "model": name, **metrics})
        if epoch < args.epochs:
            # Re-create an identically seeded loader each epoch. This guarantees
            # exactly the same shuffled batch sequence for every model.
            for name, model in models_by_name.items():
                epoch_loader = make_loader(ImageRecords(restore_records, image_transform()), args.batch_size,
                                          args.num_workers, seed + epoch)
                train_head_epoch(model, epoch_loader, optimizers[name], device, head_only=True)
    write_csv(seed_dir / "restoration_metrics.csv", rows)
    metrics = {name: [row for row in rows if row["model"] == name] for name in models_by_name}
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"seed": seed, "backbone_check": check, "metrics": metrics}


def main() -> None:
    args = parse_args()
    logger = configure_logging(args.output_dir)
    device = select_device(args.device, logger)
    frame = read_metadata(args.data_dir)
    audit_path = args.audit_manifest or args.data_dir / "counterfactual_pairs.csv"
    audit = PairedAudit(audit_path, args.data_dir, image_transform())
    global_model = load_checkpoint(args.checkpoint_global, args.architecture, args.num_classes, device)
    b_model = load_checkpoint(args.checkpoint_b, args.architecture, args.num_classes, device)
    if args.checkpoint_oracle and args.checkpoint_oracle.is_file():
        oracle_model = load_checkpoint(args.checkpoint_oracle, args.architecture, args.num_classes, device)
    elif args.train_oracle_if_missing:
        logger.warning("No oracle checkpoint supplied; training the requested balanced relation-neutral reference.")
        seed_everything(0)
        oracle_model = train_oracle(args, make_balanced_records(frame, args.data_dir), device)
    else:
        raise FileNotFoundError("M_oracle is required. Supply --checkpoint-oracle or explicitly enable --train-oracle-if-missing.")
    seed_results = [restore_one_seed(args, device, logger, global_model, b_model, oracle_model, frame, audit, seed)
                    for seed in args.seeds]
    # One wide row per seed/epoch is the requested Restoration Curve.  Per-model
    # long-form metrics remain available in each seed directory.
    curve_rows: list[dict[str, Any]] = []
    for result in seed_results:
        by_epoch = {row["epoch"]: {} for row in result["metrics"]["M_global"]}
        for model_name, model_rows in result["metrics"].items():
            for row in model_rows:
                by_epoch[row["epoch"]].update(
                    {f"{model_name}_{key}": value for key, value in row.items() if key not in {"epoch", "model"}}
                )
        curve_rows.extend({"seed": result["seed"], "epoch": epoch, **values}
                          for epoch, values in sorted(by_epoch.items()))
    write_csv(args.output_dir / "summary.csv", curve_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(seed_results, indent=2), encoding="utf-8")
    logger.info("Completed restoration experiment; outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
