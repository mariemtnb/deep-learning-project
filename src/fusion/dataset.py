"""Dataset PyTorch multimodal lu directement depuis la table produite par le pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .model import encode_tabular

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ChestXrayDataset(Dataset):
    def __init__(self, table: pd.DataFrame, image_root: Path, pathologies: list[str],
                 size: int = 224):
        self.df = table.reset_index(drop=True)
        self.root = Path(image_root)
        self.labels = [f"lbl_{p}" for p in pathologies]
        self.size = size

    def __len__(self) -> int:
        return len(self.df)

    def _image(self, row) -> torch.Tensor:
        path = self.root / row.image_folder / "images" / row.image
        with Image.open(path) as im:
            im = im.convert("L").resize((self.size, self.size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = np.repeat(arr[..., None], 3, axis=-1)          # 1 canal -> 3 (ImageNet)
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        tab = encode_tabular(row.age_clean, row.gender == "M", row.view == "AP", row.followup)
        return {
            "image": self._image(row),
            "tabular": tab,
            "tab_mask": torch.tensor(float(row.has_tabular)),
            "y": torch.tensor(row[self.labels].to_numpy(dtype=np.float32)),
        }
