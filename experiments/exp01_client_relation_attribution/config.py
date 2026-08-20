"""Locked defaults for Experiment 01.

Command-line arguments may override operational values (paths, device, rounds),
but the two relation distributions are intentionally not configurable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


GROUPS = ("WB-Water", "WB-Land", "LB-Land", "LB-Water")
SEEDS = (42, 123, 456, 789)

# Experiment-design source-pool allocation: Client A owns half of the training
# records; Clients B--E each own one eighth. FedAvg then uses the intended
# 50%/12.5% client weights through its sample-count weighting.
CLIENT_DATA_FRACTIONS = {
    0: 0.50,
    1: 0.125,
    2: 0.125,
    3: 0.125,
    4: 0.125,
}

# The only experimental manipulation.
CLIENT_A_FULL = {
    "WB-Water": 0.475,
    "WB-Land": 0.025,
    "LB-Land": 0.475,
    "LB-Water": 0.025,
}
CLIENT_NEUTRAL = {group: 0.25 for group in GROUPS}


@dataclass(frozen=True)
class ExperimentConfig:
    num_clients: int = 5
    client_a: int = 0
    global_rounds: int = 30
    local_epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 1e-3
    num_workers: int = 4
    image_size: int = 224
    audit_subjects_per_class: int = 200
    pretrained: bool = True
    freeze_backbone: bool = True
    model_name: str = "resnet50"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
