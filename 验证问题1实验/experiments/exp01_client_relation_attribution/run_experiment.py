"""CLI runner for Experiment 01: Client-A relation attribution."""

from __future__ import annotations

import argparse
import copy
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .audit import evaluate_model
from .config import ExperimentConfig, SEEDS
from .data import (
    ApproximateCounterfactualDataset,
    NaturalTestDataset,
    assignments_to_rows,
    build_paired_assignments,
    load_subject_records,
    make_client_loaders,
    stratified_client_split,
)
from .federated import train_federated
from .io_utils import hardware_info, resolve_device, set_seed, setup_logger, sha256_file, write_csv, write_json
from .model import WaterbirdsResNet, save_checkpoint
from .summarize import summarize


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--global-rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--audit-subjects-per-class", type=int, default=200)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--train-backbone", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_test and args.seeds != [42]:
        raise ValueError("Smoke test is locked to exactly --seeds 42")
    if args.smoke_test and not 8 <= args.global_rounds <= 10:
        raise ValueError("Smoke test must use 8--10 global rounds")
    if args.global_rounds < 1 or args.local_epochs < 1 or args.batch_size < 1:
        raise ValueError("Rounds, local epochs and batch size must be positive")
    return args


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        audit_subjects_per_class=args.audit_subjects_per_class,
        pretrained=not args.no_pretrained,
        freeze_backbone=not args.train_backbone,
    )


def _split_manifest(split: dict[int, list[str]]) -> list[dict[str, Any]]:
    return [
        {"client_id": client_id, "subject_id": subject_id}
        for client_id in sorted(split)
        for subject_id in split[client_id]
    ]


def _run_condition(
    condition: str,
    initial_state: dict[str, torch.Tensor],
    records: list[Any],
    assignments: dict[int, list[Any]],
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
    logger: Any,
    seed_dir: Path,
    natural_loader: DataLoader,
    audit_loader: DataLoader,
) -> dict[str, Any]:
    # Reset every stochastic source so both conditions receive the same random
    # augmentation/dropout/loader stream; only Client A's pixels differ.
    set_seed(seed)
    model = WaterbirdsResNet(config.pretrained, config.freeze_backbone).to(device)
    model.load_state_dict(initial_state, strict=True)
    loaders = make_client_loaders(
        records,
        assignments,
        config.batch_size,
        config.image_size,
        config.num_workers,
        seed,
    )
    logger.info("starting %s", condition)
    history = train_federated(
        model,
        loaders,
        config.global_rounds,
        config.local_epochs,
        config.learning_rate,
        device,
        logger,
    )
    checkpoint = seed_dir / f"{condition}.pt"
    save_checkpoint(model, str(checkpoint))
    metrics, probabilities = evaluate_model(model, natural_loader, audit_loader, device)
    write_json(seed_dir / f"{condition}_training_history.json", history)
    write_json(seed_dir / f"{condition}_probabilities.json", probabilities)
    logger.info("%s BGGap=%.6f", condition, metrics["background_gap"])
    return metrics


def run_seed(
    seed: int,
    records: list[Any],
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(seed_dir / "run.log")
    set_seed(seed)
    split = stratified_client_split(records, config.num_clients, seed)
    full_assignments, neutral_assignments = build_paired_assignments(records, split, seed)
    write_csv(seed_dir / "client_split_manifest.csv", _split_manifest(split))
    relation_rows = assignments_to_rows(full_assignments, "M_full") + assignments_to_rows(
        neutral_assignments, "M_A_neutral"
    )
    write_csv(seed_dir / "relation_assignment_manifest.csv", relation_rows)

    audit_dataset = ApproximateCounterfactualDataset(
        records,
        config.image_size,
        config.audit_subjects_per_class,
        seed=0,  # one fixed audit set across both conditions and all training seeds
    )
    write_json(seed_dir / "counterfactual_audit_manifest.json", audit_dataset.manifest())
    natural_loader = DataLoader(
        NaturalTestDataset(records, config.image_size),
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    audit_loader = DataLoader(
        audit_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    initial_model = WaterbirdsResNet(config.pretrained, config.freeze_backbone).to(device)
    initial_checkpoint = seed_dir / "initial_model.pt"
    save_checkpoint(initial_model, str(initial_checkpoint))
    initial_state = copy.deepcopy(initial_model.state_dict())
    metrics = {
        "M_full": _run_condition(
            "M_full",
            initial_state,
            records,
            full_assignments,
            config,
            seed,
            device,
            logger,
            seed_dir,
            natural_loader,
            audit_loader,
        ),
        "M_A_neutral": _run_condition(
            "M_A_neutral",
            initial_state,
            records,
            neutral_assignments,
            config,
            seed,
            device,
            logger,
            seed_dir,
            natural_loader,
            audit_loader,
        ),
    }
    full_gap = metrics["M_full"]["background_gap"]
    neutral_gap = metrics["M_A_neutral"]["background_gap"]
    metrics["paired_comparison"] = {
        "contribution": full_gap - neutral_gap,
        "relative_reduction": (full_gap - neutral_gap) / full_gap if full_gap > 0 else None,
        "direction_consistent": neutral_gap < full_gap,
        "signed_deltas_toward_zero": {
            "waterbird": abs(metrics["M_A_neutral"]["delta_waterbird"]) < abs(metrics["M_full"]["delta_waterbird"]),
            "landbird": abs(metrics["M_A_neutral"]["delta_landbird"]) < abs(metrics["M_full"]["delta_landbird"]),
        },
    }
    write_json(seed_dir / "metrics.json", metrics)
    write_json(
        seed_dir / "artifact_hashes.json",
        {
            path.name: sha256_file(path)
            for path in (
                initial_checkpoint,
                seed_dir / "M_full.pt",
                seed_dir / "M_A_neutral.pt",
                seed_dir / "client_split_manifest.csv",
                seed_dir / "relation_assignment_manifest.csv",
                seed_dir / "counterfactual_audit_manifest.json",
            )
        },
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    config = _config_from_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "experiment": "exp01_client_relation_attribution",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir),
        "seeds": args.seeds,
        "smoke_test": args.smoke_test,
        "config": config.to_dict(),
        "hardware": hardware_info(device),
        "intervention": "Client 0 composite-image group sampling distribution",
        "evidence_scope": "approximate distribution intervention; not strict same-subject background replacement",
    }
    write_json(output_dir / "experiment_info.json", info)
    print(f"PyTorch version: {info['hardware']['torch_version']}", flush=True)
    print(f"CUDA available: {info['hardware']['cuda_available']}", flush=True)
    print(f"CUDA version: {info['hardware']['cuda_version']}", flush=True)
    print(f"GPU name: {info['hardware']['gpu_name']}", flush=True)
    records = load_subject_records(args.data_dir.resolve())
    all_results = {
        seed: run_seed(seed, records, output_dir, config, device)
        for seed in args.seeds
    }
    summarize(all_results, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
