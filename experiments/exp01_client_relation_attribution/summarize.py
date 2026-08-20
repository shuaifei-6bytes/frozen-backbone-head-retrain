"""Cross-seed paired summaries and preregistered directional checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .io_utils import write_csv, write_json


SCALAR_METRICS = (
    "delta_waterbird",
    "delta_landbird",
    "background_gap",
    "original_flip_rate",
    "reverse_flip_rate",
    "overall_accuracy",
    "worst_group_accuracy",
)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "mean_plus_minus_std": f"{array.mean():.6f} ± {(array.std(ddof=1) if len(array) > 1 else 0.0):.6f}",
        "values": [float(value) for value in array],
    }


def summarize(seed_results: dict[int, dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    conditions = ("M_full", "M_A_neutral")
    summary: dict[str, Any] = {condition: {} for condition in conditions}
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for metric in SCALAR_METRICS:
            values = [seed_results[seed][condition][metric] for seed in sorted(seed_results)]
            stats = _stats(values)
            summary[condition][metric] = stats
            rows.append({"condition": condition, "metric": metric, "mean": stats["mean"], "std": stats["std"]})
        for group in seed_results[next(iter(seed_results))][condition]["group_accuracies"]:
            values = [seed_results[seed][condition]["group_accuracies"][group] for seed in sorted(seed_results)]
            stats = _stats(values)
            summary[condition].setdefault("group_accuracies", {})[group] = stats
            rows.append({"condition": condition, "metric": f"group_accuracy.{group}", "mean": stats["mean"], "std": stats["std"]})

    contributions = []
    for seed in sorted(seed_results):
        full_gap = float(seed_results[seed]["M_full"]["background_gap"])
        neutral_gap = float(seed_results[seed]["M_A_neutral"]["background_gap"])
        absolute = full_gap - neutral_gap
        relative = absolute / full_gap if full_gap > 0 else None
        contributions.append(
            {
                "seed": seed,
                "background_gap_full": full_gap,
                "background_gap_neutral": neutral_gap,
                "contribution": absolute,
                "relative_reduction": relative,
                "direction_consistent": neutral_gap < full_gap,
                "gate_1_full_relation_present": (
                    full_gap > 0.15
                    and seed_results[seed]["M_full"]["delta_waterbird"] > 0.0
                    and seed_results[seed]["M_full"]["delta_landbird"] > 0.0
                ),
                "gate_2_reduction_at_least_40_percent": (
                    neutral_gap < full_gap and relative is not None and relative >= 0.40
                ),
                # No arbitrary magnitude threshold is introduced: a negative
                # signed delta is already evidence of the opposite shortcut.
                "gate_3_no_signed_relation_reversal": (
                    seed_results[seed]["M_A_neutral"]["delta_waterbird"] >= 0.0
                    and seed_results[seed]["M_A_neutral"]["delta_landbird"] >= 0.0
                ),
            }
        )
    relative_values = [row["relative_reduction"] for row in contributions if row["relative_reduction"] is not None]
    summary["paired_contribution"] = {
        "per_seed": contributions,
        "contribution": _stats([row["contribution"] for row in contributions]),
        "relative_reduction": _stats(relative_values) if relative_values else None,
        "direction_consistent_count": sum(row["direction_consistent"] for row in contributions),
        "direction_consistent_required": 3,
        "at_least_3_of_4": len(contributions) == 4 and sum(row["direction_consistent"] for row in contributions) >= 3,
        "mean_relative_reduction_at_least_40_percent": bool(relative_values and np.mean(relative_values) >= 0.40),
        "pre_registered_gates": {
            "all_seeds_full_relation_present": all(row["gate_1_full_relation_present"] for row in contributions),
            "all_seeds_reduction_at_least_40_percent": all(
                row["gate_2_reduction_at_least_40_percent"] for row in contributions
            ),
            "all_seeds_no_signed_relation_reversal": all(
                row["gate_3_no_signed_relation_reversal"] for row in contributions
            ),
        },
    }
    # Keep the design-sheet label as a compact, stable entry point while the
    # richer paired_contribution block retains all per-seed diagnostics.
    summary["Contribution"] = {
        "absolute": summary["paired_contribution"]["contribution"],
        "relative": summary["paired_contribution"]["relative_reduction"],
    }
    full_mean_gap = summary["M_full"]["background_gap"]["mean"]
    summary["interpretation_guard"] = {
        "relation_injection_check_required": True,
        "M_full_mean_background_gap": full_mean_gap,
        "note": "If M_full BGGap is low, report relation injection failure before interpreting Client A contribution.",
    }
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "summary.csv", rows)
    return summary
