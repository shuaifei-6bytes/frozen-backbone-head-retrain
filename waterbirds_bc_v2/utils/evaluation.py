"""
Waterbirds B/C 实验评估工具：四组准确率、signed effect、Background Gap、方向翻转率
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.model import ResNetWithHead
import config.config as cfg

GROUP_NAMES = cfg.GROUP_NAMES


class Evaluator:
    """对单个模型做四组准确率 + 反事实背景依赖指标评估。"""

    def __init__(self, model: ResNetWithHead, device: torch.device = cfg.DEVICE):
        self.model = model
        self.device = device

    def evaluate_group_accuracy(self, data_loader: DataLoader) -> dict:
        """按四 group 分别计算准确率。"""
        correct = {g: 0 for g in GROUP_NAMES}
        total = {g: 0 for g in GROUP_NAMES}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="group acc"):
                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                groups = batch["group"]
                _, pred = self.model(imgs).max(1)
                for i, g in enumerate(groups):
                    total[g] += 1
                    if pred[i] == labels[i]:
                        correct[g] += 1
        return {g: (100.0 * correct[g] / total[g] if total[g] else 0.0) for g in GROUP_NAMES}

    def evaluate_overall_accuracy(self, data_loader: DataLoader) -> float:
        """整体准确率。"""
        correct = total = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="overall acc"):
                imgs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                _, pred = self.model(imgs).max(1)
                total += labels.size(0)
                correct += pred.eq(labels).sum().item()
        return 100.0 * correct / total

    def evaluate_counterfactual(self, counterfactual_loader: DataLoader) -> dict:
        """逐反事实对输出目标类概率，供 signed effect/翻转率推导。

        对每个 pair：
          - kind=waterbird: imgA=水背景样本, imgB=陆地背景样本；目标类=水鸟(下标0)
          - kind=landbird : imgA=陆地背景样本, imgB=水背景样本；目标类=陆鸟(下标1)
        pA/pB 为对应目标类概率。
        """
        self.model.eval()
        out = {"kind": [], "pA": [], "pB": []}
        with torch.no_grad():
            for batch in tqdm(counterfactual_loader, desc="counterfactual"):
                imgA = batch["imgA"].to(self.device)
                imgB = batch["imgB"].to(self.device)
                kinds = batch["kind"]
                pa = F.softmax(self.model(imgA), dim=1)
                pb = F.softmax(self.model(imgB), dim=1)
                for i, k in enumerate(kinds):
                    target_idx = 0 if k == "waterbird" else 1
                    out["kind"].append(k)
                    out["pA"].append(pa[i][target_idx].item())
                    out["pB"].append(pb[i][target_idx].item())
        return out

    def compute_background_metrics(self, cf: dict) -> dict:
        """由反事实对计算 signed effect、Background Gap 与方向翻转率，并做一致性校验。"""
        kinds = np.array(cf["kind"])
        pA = np.array(cf["pA"])
        pB = np.array(cf["pB"])

        wb = kinds == "waterbird"      # 水鸟对
        lb = kinds == "landbird"       # 陆鸟对

        # signed effect（文档 §8）
        delta_waterbird = float(np.mean(pA[wb] - pB[wb]))
        delta_landbird = float(np.mean(pA[lb] - pB[lb]))

        # Background Gap = (|ΔWB| + |ΔLB|) / 2（文档 §9）
        bg_gap = (abs(delta_waterbird) + abs(delta_landbird)) / 2

        # 逐对方向翻转（文档 §10）：目标类概率 >0.5 判为目标，<0.5 判为另一类
        # original: 主背景判目标类，另一背景判反类
        # reverse : 主背景判反类，另一背景判目标类
        all_orig = (pA > 0.5) & (pB < 0.5)
        all_rev = (pA < 0.5) & (pB > 0.5)
        n_total = len(kinds)
        original_flip_rate = float(all_orig.sum() / n_total)
        reverse_flip_rate = float(all_rev.sum() / n_total)

        # 一致性校验（文档 §13/14）：Δ 必须由底层概率严格推出
        assert abs(bg_gap - ((abs(delta_waterbird) + abs(delta_landbird)) / 2)) < 1e-9, \
            "Background Gap 与 signed effect 不一致"
        assert abs(delta_waterbird - float(np.mean(pA[wb] - pB[wb]))) < 1e-9
        assert abs(delta_landbird - float(np.mean(pA[lb] - pB[lb]))) < 1e-9

        return {
            "delta_waterbird": delta_waterbird,
            "delta_landbird": delta_landbird,
            "background_gap": bg_gap,
            "original_flip_rate": original_flip_rate,
            "reverse_flip_rate": reverse_flip_rate,
        }

    def evaluate_model(self, data_loader: DataLoader,
                       counterfactual_loader: DataLoader,
                       model_name: str, save_dir: str) -> dict:
        """综合评估一个模型并落盘 metrics / probabilities。"""
        group_acc = self.evaluate_group_accuracy(data_loader)
        results = {
            "model_name": model_name,
            "overall_accuracy": self.evaluate_overall_accuracy(data_loader),
            "worst_group_accuracy": min(group_acc.values()),
            "group_accuracies": group_acc,
        }
        cf = self.evaluate_counterfactual(counterfactual_loader)
        results.update(self.compute_background_metrics(cf))

        # 保存底层概率（文档 §13），支持后续审计
        kinds = np.array(cf["kind"])
        pA = np.array(cf["pA"])
        pB = np.array(cf["pB"])
        wb = kinds == "waterbird"
        lb = kinds == "landbird"
        probabilities = {
            # §13 四个底层概率：P_WB_WATER / P_WB_LAND / P_LB_LAND / P_LB_WATER
            "P_WB_WATER": pA[wb].tolist(),   # 水鸟 + 水背景 → 判水鸟概率
            "P_WB_LAND": pB[wb].tolist(),    # 水鸟 + 陆地背景 → 判水鸟概率
            "P_LB_LAND": pA[lb].tolist(),    # 陆鸟 + 陆地背景 → 判陆鸟概率
            "P_LB_WATER": pB[lb].tolist(),   # 陆鸟 + 水背景 → 判陆鸟概率
            "n_waterbird_pairs": int(wb.sum()),
            "n_landbird_pairs": int(lb.sum()),
        }

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{model_name}_metrics.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, f"{model_name}_probabilities.json"), "w") as f:
            json.dump(probabilities, f, indent=2, ensure_ascii=False)
        return results
