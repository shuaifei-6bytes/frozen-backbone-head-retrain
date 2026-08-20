"""Run Experiment 2: relation neutralization versus deleting Client A.

Example (Kaggle/Colab):
python run_experiment.py --device cuda --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/exp02_results --seeds 42 123 --global-rounds 15
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean, stdev

import torch

from src.experiment import (
    FULL_A, NEUTRAL, ImageDataset, build_source_pools, evaluate, evaluation_sets,
    fedavg, loaders, load_records, make_model, manifest_rows, relation_metrics,
    sample_assignments, seed_everything, write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--global-rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1, choices=(1,))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160, choices=(160,))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available(): raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def metric_bundle(model, records, natural_test, a_holdout, be_holdouts, args, device):
    result = evaluate(model, natural_test, args.batch_size, device)
    result.update(relation_metrics(model, records, args.image_size, args.batch_size, device))
    result["client_a_neutral_accuracy"] = evaluate(model, a_holdout, args.batch_size, device)["overall_accuracy"]
    client_scores = {str(client): evaluate(model, dataset, args.batch_size, device)["overall_accuracy"] for client, dataset in be_holdouts.items()}
    result["be_client_accuracies"] = client_scores
    result["be_mean_accuracy"] = mean(client_scores.values())
    result["worst_client_accuracy"] = min(client_scores.values())
    return result


def run_seed(seed: int, records, args, device: torch.device) -> dict:
    seed_everything(seed)
    out = args.output_dir / f"seed_{seed}"; out.mkdir(parents=True, exist_ok=True)
    pools = build_source_pools(records, seed)
    full = sample_assignments(pools, seed, FULL_A)
    neutral = sample_assignments(pools, seed, NEUTRAL)
    # Identical B--E manifests are a non-negotiable paired-control assertion.
    if any(full[i] != neutral[i] for i in range(1, 5)): raise AssertionError("B--E assignments changed between paired conditions")
    with (out / "relation_assignment_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows(full, "M_full")[0]))
        writer.writeheader(); writer.writerows(manifest_rows(full, "M_full")); writer.writerows(manifest_rows(neutral, "M_A_neutral"))
    # One initial checkpoint is serialized, then used byte-for-byte in all three arms.
    initial = make_model().cpu().state_dict(); torch.save(initial, out / "init_model.pt")
    natural_test = ImageDataset([r for r in records if r.split == "test"], args.image_size)
    a_holdout, be_holdouts = evaluation_sets(records, seed)
    conditions = {
        "M_full": loaders(full, args.batch_size, args.image_size, args.num_workers, seed),
        "M_A_neutral": loaders(neutral, args.batch_size, args.image_size, args.num_workers, seed),
        "M_minus_A": loaders({i: full[i] for i in range(1, 5)}, args.batch_size, args.image_size, args.num_workers, seed),
    }
    models, metrics, histories = {}, {}, {}
    for name, client_loaders in conditions.items():
        model, history = fedavg(initial, client_loaders, args.global_rounds, args.lr, device)
        models[name], histories[name] = model, history
        torch.save(model.cpu().state_dict(), out / f"{name}.pt"); model.to(device)
        metrics[name] = metric_bundle(model, records, natural_test, a_holdout, be_holdouts, args, device)
    full_metrics = metrics["M_full"]
    over = {}
    for arm in ("M_A_neutral", "M_minus_A"):
        over[arm] = {
            "overall_accuracy_loss": full_metrics["overall_accuracy"] - metrics[arm]["overall_accuracy"],
            "client_a_normal_knowledge_loss": full_metrics["client_a_neutral_accuracy"] - metrics[arm]["client_a_neutral_accuracy"],
            "be_utility_loss": full_metrics["be_mean_accuracy"] - metrics[arm]["be_mean_accuracy"],
        }
    gates = {
        "gate_1_relation_weakened": all(metrics[a]["background_gap"] < full_metrics["background_gap"] for a in ("M_A_neutral", "M_minus_A")),
        "gate_2_neutral_retains_more": metrics["M_A_neutral"]["overall_accuracy"] > metrics["M_minus_A"]["overall_accuracy"],
        "gate_2_at_least_one_margin": any((metrics["M_A_neutral"][k] - metrics["M_minus_A"][k]) >= margin for k, margin in (("overall_accuracy", .02), ("worst_group_accuracy", .05), ("client_a_neutral_accuracy", .05))),
        "gate_3_be_mean_loss_within_2pp": full_metrics["be_mean_accuracy"] - metrics["M_A_neutral"]["be_mean_accuracy"] <= .02,
    }
    payload = {"seed": seed, "metrics": metrics, "over_forgetting": over, "gates": gates, "training_history": histories}
    write_json(out / "metrics.json", payload)
    write_json(out / "counterfactual_scope.json", {"strict_same_subject_background_swap_available": False, "status": "approximate_same_label_cross_image_background pairs", "reason": "Standard Waterbirds composites expose no alternative background rendering for an image id. Training keeps Client-A source pool and sample count fixed, but neutral group resampling cannot retain the exact occurrence image-id set."})
    return payload


def summarize(results: list[dict], output: Path) -> None:
    scalar_keys = ("overall_accuracy", "worst_group_accuracy", "client_a_neutral_accuracy", "be_mean_accuracy", "worst_client_accuracy", "delta_waterbird", "delta_landbird", "background_gap", "original_flip_rate", "reverse_flip_rate")
    summary = {}
    for arm in ("M_full", "M_A_neutral", "M_minus_A"):
        summary[arm] = {k: {"mean": mean([r["metrics"][arm][k] for r in results]), "std": stdev([r["metrics"][arm][k] for r in results]) if len(results)>1 else 0.0} for k in scalar_keys}
    over = {arm: {k: {"mean": mean([r["over_forgetting"][arm][k] for r in results]), "std": stdev([r["over_forgetting"][arm][k] for r in results]) if len(results)>1 else 0.0} for k in results[0]["over_forgetting"][arm]} for arm in ("M_A_neutral", "M_minus_A")}
    write_json(output / "summary.json", {"models": summary, "over_forgetting": over})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle); writer.writerow(["metric", "M_full_mean", "M_full_std", "M_A_neutral_mean", "M_A_neutral_std", "M_minus_A_mean", "M_minus_A_std"])
        for metric in scalar_keys: writer.writerow([metric, *sum(([summary[a][metric][x] for x in ("mean", "std")] for a in ("M_full", "M_A_neutral", "M_minus_A")), [])])


def main() -> int:
    args = arguments(); args.output_dir.mkdir(parents=True, exist_ok=True); device = device_for(args.device)
    logging.basicConfig(filename=args.output_dir / "run.log", level=logging.INFO, format="%(asctime)s %(message)s")
    info = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "device": str(device), "config": vars(args), "design_limitations": "See per-seed counterfactual_scope.json; no same-subject background swaps exist in standard composite Waterbirds."}
    info["config"]["data_dir"] = str(args.data_dir); info["config"]["output_dir"] = str(args.output_dir); write_json(args.output_dir / "experiment_info.json", info)
    print(json.dumps(info, indent=2, default=str)); records = load_records(args.data_dir.resolve())
    results = [run_seed(seed, records, args, device) for seed in args.seeds]
    summarize(results, args.output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
