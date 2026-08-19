"""
Waterbirds 数据集与反事实评估对工具（对接真实 waterbird_complete95 数据）
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# 全局常量：group 四分类与二分类 label 的对应关系
# waterbird: 1=水鸟 0=陆鸟；water_background: 1=水背景 0=陆地背景
GROUP_OF = {
    (1, 1): "WB-Water",   # 水鸟 + 水背景
    (1, 0): "WB-Land",    # 水鸟 + 陆地背景
    (0, 0): "LB-Land",    # 陆鸟 + 陆地背景
    (0, 1): "LB-Water",   # 陆鸟 + 水背景
}
GROUP_NAMES = ["WB-Water", "WB-Land", "LB-Land", "LB-Water"]


def load_metadata(root_dir: str, split: str = "train") -> pd.DataFrame:
    """读取 waterbirds metadata.csv 并过滤指定 split，归一化列名。

    root_dir 指向 waterbird_complete95_forest2water2（含 metadata.csv 与图片子目录）。
    兼容两种元数据格式：
      - 标准：waterbird / water_background / img_path / split(train|val|test)
      - Kaggle 简版：y / place / img_filename / split(0|1|2)
    统一映射为 waterbird(1=水鸟)、water_background(1=水背景)、_fullpath、并按 split 过滤。
    """
    meta = pd.read_csv(os.path.join(root_dir, "metadata.csv"))

    # split 过滤：字符串或数值(0=train,1=val,2=test)两种编码
    if "split" in meta.columns:
        if meta["split"].dtype != object:
            idx = {"train": 0, "val": 1, "test": 2}[split]
            meta = meta[meta["split"].astype(int) == idx]
        else:
            meta = meta[meta["split"] == split]

    # 图片路径列（标准 img_path / 简版 img_filename，均含子目录，相对 root）
    if "img_path" in meta.columns:
        img_col = "img_path"
    elif "img_filename" in meta.columns:
        img_col = "img_filename"
    else:
        raise KeyError(f"metadata.csv 缺少图片路径列，现有列: {list(meta.columns)}")
    meta["_fullpath"] = meta[img_col].apply(
        lambda p: p if os.path.isabs(str(p)) else os.path.join(root_dir, str(p))
    )

    # 鸟类别列：waterbird=1 水鸟 / 0 陆鸟；简版用 y
    if "waterbird" not in meta.columns and "y" in meta.columns:
        meta["waterbird"] = meta["y"]

    # 背景列：water_background=1 水背景 / 0 陆地背景；简版用 place
    if "water_background" not in meta.columns and "place" in meta.columns:
        meta["water_background"] = meta["place"]

    for required in ("waterbird", "water_background"):
        if required not in meta.columns:
            raise KeyError(f"metadata.csv 缺少列 {required}，现有列: {list(meta.columns)}")

    return meta.reset_index(drop=True)


def _group_series(meta: pd.DataFrame):
    """把 waterbird / water_background 两列映射为 group 名列表。"""
    wb = meta["waterbird"].values
    bg = meta["water_background"].values
    return [GROUP_OF[(int(w), int(b))] for w, b in zip(wb, bg)]


def get_transform(split: str):
    """训练用随机水平翻转；评估用无增强。均 224 归一化。"""
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if split == "train":
        base.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)


class WaterbirdsDataset(Dataset):
    """真实 Waterbirds 数据集。

    支持按 group 分布采样：不足目标数量的 group 会用重复采样补齐，
    保证 B/C（以及全局训练）每个 epoch 看到大致相同的样本总量（§6 公平性）。
    """

    def __init__(self, root_dir: str, split: str = "train",
                 distribution: Dict[str, float] = None,
                 transform=None, seed: int = 42):
        self.split = split
        self.distribution = distribution or {}
        self.seed = seed
        self.transform = transform or get_transform(split)

        meta = load_metadata(root_dir, split)
        self.meta = meta
        self.groups = _group_series(meta)
        self.labels = meta["waterbird"].astype(int).tolist()

        # 采样后保留的原始行下标（用于 __getitem__）
        self.indices = self._sample_indices()

    def _sample_indices(self):
        """根据 distribution 生成每个 group 的抽样下标；不足则重复补齐。"""
        if not self.distribution:
            return list(range(len(self.meta)))

        # 按 group 分组原始下标
        grouped: Dict[str, List[int]] = {g: [] for g in GROUP_NAMES}
        for i, g in enumerate(self.groups):
            grouped[g].append(i)

        total = len(self.meta)
        rng = np.random.RandomState(self.seed)
        picked: List[int] = []
        for group in GROUP_NAMES:
            if group not in self.distribution or not grouped[group]:
                continue
            idx_pool = np.array(grouped[group])
            target = int(self.distribution[group] * total)
            if target <= 0:
                continue
            # 目标超过实际数量 → 可重复采样补齐，使各 epoch 样本量对齐
            replace = target > len(idx_pool)
            sel = rng.choice(idx_pool, size=target, replace=replace)
            picked.extend(sel.tolist())
        return picked

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.meta.iloc[self.indices[idx]]
        img = Image.open(row["_fullpath"]).convert("RGB")
        img = self.transform(img)
        return {
            "image": img,
            "label": torch.tensor(self.labels[self.indices[idx]], dtype=torch.long),
            "group": self.groups[self.indices[idx]],
            "sample_id": str(self.indices[idx]),
        }


class CounterfactualDataset(Dataset):
    """反事实评估对数据集。

    对每个主体类型构造"同一类鸟主体 + 两种背景"的配对：
      - 水鸟对：水背景样本 与 陆地背景样本 配对
      - 陆鸟对：陆地背景样本 与 水背景样本 配对
    配对数量取两 group 样本数的较小值，用固定 seed 随机配对以保证可复现。
    ponytail: 真实数据没有"同一只鸟"的两种背景，这里用同类鸟随机配对近似主体不变；
    要做严格同一主体需背景合成，成本高，暂不做。
    """

    def __init__(self, root_dir: str, split: str = "test",
                 transform=None, num_pairs: int = 200, seed: int = 0):
        self.transform = transform or get_transform("val")
        meta = load_metadata(root_dir, split)
        groups = _group_series(meta)
        self.labels = meta["waterbird"].astype(int).tolist()
        self.paths = meta["_fullpath"].tolist()

        rng = np.random.RandomState(seed)
        self.pairs = []  # 每项: (kind, imgA_path, imgB_path)
        for kind, gA, gB in [("waterbird", "WB-Water", "WB-Land"),
                             ("landbird", "LB-Land", "LB-Water")]:
            idxA = [i for i, g in enumerate(groups) if g == gA]
            idxB = [i for i, g in enumerate(groups) if g == gB]
            k = min(len(idxA), len(idxB), num_pairs)
            if k == 0:
                continue
            pickA = rng.choice(idxA, size=k, replace=False)
            pickB = rng.choice(idxB, size=k, replace=False)
            for a, b in zip(pickA, pickB):
                self.pairs.append((kind, self.paths[a], self.paths[b]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        kind, pathA, pathB = self.pairs[idx]
        imgA = self.transform(Image.open(pathA).convert("RGB"))
        imgB = self.transform(Image.open(pathB).convert("RGB"))
        return {"kind": kind, "imgA": imgA, "imgB": imgB}


def create_data_loaders(data_dir: str, batch_size: int = 32,
                        distribution: Dict[str, float] = None,
                        split: str = "train",
                        seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """创建训练集（可带分布采样）与全量验证集。"""
    train_dataset = WaterbirdsDataset(
        root_dir=data_dir, split="train",
        distribution=distribution, transform=get_transform("train"), seed=seed,
    )
    # 验证集始终全量（不套用训练分布），保证评估口径干净
    val_dataset = WaterbirdsDataset(
        root_dir=data_dir, split="val", transform=get_transform("val"), seed=seed,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


def create_counterfactual_loader(data_dir: str, transform=None,
                                 num_pairs: int = 200, batch_size: int = 64,
                                 seed: int = 0) -> DataLoader:
    """构造反事实评估数据加载器。"""
    dataset = CounterfactualDataset(data_dir, split="test", transform=transform,
                                    num_pairs=num_pairs, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)
