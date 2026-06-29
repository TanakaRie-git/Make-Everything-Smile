"""PyTorch Dataset for InstructPix2Pix smile-transfer training.

`build_pairs.py` が出力した JSON（各要素が original/target/instruction を持つ）を読み、
[-1, 1] に正規化した pixel tensor と instruction 文字列を返す。

返り値（__getitem__）:
    {
        "original_pixel_values": FloatTensor[3, H, W]  # 入力(非笑顔), [-1,1]
        "target_pixel_values":   FloatTensor[3, H, W]  # 教師(笑顔),   [-1,1]
        "instruction":           str
    }

注意: original と target は別人物（同一人物ペアが存在しないため）。
学習時の random flip は original/target を独立に反転させると整合しないが、
そもそも別人物なので幾何的整合は不要。ここでは安全側で同一の flip を両者に適用する。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SmilePairDataset(Dataset):
    def __init__(self, pairs_json: str | Path, resolution: int = 256,
                 random_flip: bool = True):
        self.pairs_json = Path(pairs_json)
        if not self.pairs_json.is_file():
            raise FileNotFoundError(f"pairs JSON が見つかりません: {self.pairs_json}")
        with self.pairs_json.open("r", encoding="utf-8") as f:
            self.pairs = json.load(f)
        if len(self.pairs) == 0:
            raise ValueError(f"pairs JSON が空です: {self.pairs_json}")

        self.resolution = resolution
        self.random_flip = random_flip

        # [-1, 1] 正規化（VAE の入力仕様に合わせる）
        self.resize = transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR
        )
        self.crop = transforms.CenterCrop(resolution)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.pairs)

    def _load(self, path: str, flip: bool) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = self.crop(self.resize(img))
        if flip:
            img = transforms.functional.hflip(img)
        return self.normalize(self.to_tensor(img))

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        flip = self.random_flip and (torch.rand(1).item() < 0.5)
        return {
            "original_pixel_values": self._load(item["original"], flip),
            "target_pixel_values": self._load(item["target"], flip),
            "instruction": item.get("instruction", "make the person smile"),
        }


def collate_fn(batch):
    """DataLoader 用。instruction は文字列リストのまま渡す（後段で tokenize する）。"""
    return {
        "original_pixel_values": torch.stack(
            [b["original_pixel_values"] for b in batch]
        ),
        "target_pixel_values": torch.stack(
            [b["target_pixel_values"] for b in batch]
        ),
        "instruction": [b["instruction"] for b in batch],
    }
