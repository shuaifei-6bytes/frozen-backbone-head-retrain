"""
Configuration for Waterbirds B/C experiment V2
"""

import torch
import os
from typing import Dict, List, Tuple

# Basic device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

# Fixed seeds
SEEDS = [42, 123, 456, 789]

# Dataset configuration
DATASET_NAME = "waterbirds"
NUM_GROUPS = 4
GROUP_NAMES = ["WB-Water", "WB-Land", "LB-Land", "LB-Water"]

# Federal learning configuration
NUM_CLIENTS = 5
GLOBAL_ROUNDS = 30
IID_CLIENTS = True
ALL_CLIENTS_PARTICIPATE = True

# Data distribution for global training
GLOBAL_TRAIN_DISTRIBUTION = {
    "WB-Water": 0.475,      # 47.5% water birds + water background
    "WB-Land": 0.025,       # 2.5% water birds + land background  
    "LB-Land": 0.475,       # 47.5% land birds + land background
    "LB-Water": 0.025       # 2.5% land birds + water background
}

# Model configuration
MODEL_NAME = "resnet50"
PRETRAINED = True
FREEZE_BACKBONE = True

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 3   # 每个客户端本地 epoch，从 1 提到 3：保证 head 在冻结 backbone 下学出区分度

# Head retraining configuration
HEAD_RETRAIN_EPOCHS = 20
HEAD_RETRAIN_BATCH_SIZE = 32
HEAD_RETRAIN_LR = 0.001

# Balanced Head Retraining (B) configuration
B_DISTRIBUTION = {
    "WB-Water": 0.25,      # 25% water birds + water background
    "WB-Land": 0.25,       # 25% water birds + land background  
    "LB-Land": 0.25,       # 25% land birds + land background
    "LB-Water": 0.25       # 25% land birds + water background
}

# Conflicting-only Head Retraining (C) configuration  
C_DISTRIBUTION = {
    "WB-Land": 0.5,        # 50% water birds + land background
    "LB-Water": 0.5        # 50% land birds + water background
}

# Evaluation configuration
EVAL_BATCH_SIZE = 64
SHUFFLE_EVAL = False
NO_AUGMENTATION = True

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")

def _resolve_data_dir() -> str:
    """定位含 metadata.csv 的 waterbirds 数据目录。

    优先用写死的 Kaggle 完整路径；不存在时自动扫描 /kaggle/input 找 metadata.csv，
    兼顾 Kaggle 数据集挂载层级差异。
    """
    cand = "/kaggle/input/datasets/feishuai/waterbird-complete95/waterbird_complete95_forest2water2"
    if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "metadata.csv")):
        return cand
    for root, _, files in os.walk("/kaggle/input"):
        if "metadata.csv" in files:
            return root
    return cand

DATA_DIR = _resolve_data_dir()

# 反事实评估对数量（每类主体）
NUM_COUNTERFACTUAL_PAIRS = 200

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)

# Metrics to track
METRICS = [
    "overall_accuracy",
    "worst_group_accuracy", 
    "delta_waterbird",
    "delta_landbird",
    "background_gap",
    "original_flip_rate",
    "reverse_flip_rate",
    "group_accuracies"
]

# Probability names to save
PROBABILITY_NAMES = [
    "P_WB_WATER",  # P(waterbird | waterbird + water background)
    "P_WB_LAND",   # P(waterbird | waterbird + land background)
    "P_LB_LAND",   # P(landbird | landbird + land background)
    "P_LB_WATER"   # P(landbird | landbird + water background)
]