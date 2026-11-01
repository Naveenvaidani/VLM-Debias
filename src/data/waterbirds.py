"""
Waterbirds (Sagawa et al., 2020) spurious-correlation benchmark.

Label y: 0 = landbird, 1 = waterbird. Spurious attribute place: 0 = land, 1 = water.
In the training split 95% of birds appear on their "matching" background, so the
background is a shortcut. group = 2 * y + place:
    0 landbird/land   (majority)    1 landbird/water (minority)
    2 waterbird/land  (minority)    3 waterbird/water (majority)
"""
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = {"train": 0, "val": 1, "test": 2}
GROUP_NAMES = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
MAJORITY_GROUPS = (0, 3)
MINORITY_GROUPS = (1, 2)
CLASS_NAMES = ["landbird", "waterbird"]


def find_waterbirds_root(data_dir: str = "./data") -> str:
    for name in sorted(os.listdir(data_dir)):
        root = os.path.join(data_dir, name)
        if os.path.isfile(os.path.join(root, "metadata.csv")):
            return root
    raise FileNotFoundError(f"Waterbirds metadata.csv not found under {data_dir}")


def load_metadata(root: str) -> Dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(os.path.join(root, "metadata.csv"))
    return {
        "img_filename": df["img_filename"].to_numpy(),
        "y": df["y"].to_numpy().astype(np.int64),
        "place": df["place"].to_numpy().astype(np.int64),
        "split": df["split"].to_numpy().astype(np.int64),
        "group": (2 * df["y"] + df["place"]).to_numpy().astype(np.int64),
    }


class CachedFeatureDataset(Dataset):
    """
    Serves frozen SigLIP patch features precomputed by scripts/cache_siglip_features.py.
    Features live in a single memory-mapped fp16 array [N, num_patches, vision_dim].
    """
    def __init__(self, feature_path: str, meta: Dict[str, np.ndarray], split: str):
        self.features = np.load(feature_path, mmap_mode="r")
        self.indices = np.nonzero(meta["split"] == SPLITS[split])[0]
        self.y = meta["y"][self.indices]
        self.group = meta["group"][self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        feats = torch.from_numpy(np.array(self.features[idx], dtype=np.float32))
        return feats, int(self.y[i]), int(self.group[i])


def group_metrics(preds: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    """Average, per-group, worst-group, majority (ID) and minority (OOD) accuracy."""
    correct = preds == labels
    out = {"avg_acc": float(correct.mean())}
    group_accs: List[float] = []
    for g, name in enumerate(GROUP_NAMES):
        mask = groups == g
        acc = float(correct[mask].mean()) if mask.any() else float("nan")
        out[f"acc_{name}"] = acc
        group_accs.append(acc)
    out["worst_group_acc"] = float(np.nanmin(group_accs))
    out["acc_id"] = float(correct[np.isin(groups, MAJORITY_GROUPS)].mean())
    out["acc_ood"] = float(correct[np.isin(groups, MINORITY_GROUPS)].mean())
    out["delta_ood"] = out["acc_id"] - out["acc_ood"]
    out["rsr"] = out["delta_ood"] / max(out["acc_id"], 1e-8)
    return out
