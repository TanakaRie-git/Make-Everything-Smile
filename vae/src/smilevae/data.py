"""Datasets and the shared image normalization used across the project.

Plan §2.2: object-centered crop, aspect-preserving resize with white padding
(matching the Faces in Things convention), fixed resolution.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pad_to_square_white(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def build_transform(resolution: int, train: bool = False) -> transforms.Compose:
    ops = [
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Lambda(pad_to_square_white),
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1] to match tanh output
    ]
    return transforms.Compose(ops)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


class FlatImageDataset(Dataset):
    """All images found under one or more directories, no labels."""

    def __init__(self, dirs: list[str | Path], resolution: int, train: bool = True):
        self.paths: list[Path] = []
        for d in dirs:
            found = list_images(Path(d))
            if not found:
                raise FileNotFoundError(f"no images under {d}")
            self.paths += found
        self.transform = build_transform(resolution, train=train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]))
