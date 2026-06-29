"""Datasets for the GAN smile-editing track.

Each training item is (image, smile_label, is_face):
  smile_label: 1.0 for smiling faces, 0.0 for neutral faces and objects.
  is_face: True for CelebA images; False for ImageNet/Imagenette objects.
  The is_face flag lets the training loop apply D's smile regression loss only
  to face images, which carry ground-truth smile annotations (plan §2.1).

Plan §2.2 normalization: aspect-preserving resize with white padding → [-1, 1].
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
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
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # → [-1, 1] to match tanh output
    ]
    return transforms.Compose(ops)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTENSIONS)


class FlatImageDataset(Dataset):
    """All images under one or more directories, no labels. Used by direction.py and edit.py."""

    def __init__(self, dirs: list[str | Path], resolution: int, train: bool = False):
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


class _LabeledDataset(Dataset):
    """Images from a set of directories, all assigned the same fixed label."""

    def __init__(self, dirs: list[str | Path], label: float, is_face: bool,
                 resolution: int, train: bool = True):
        self.paths: list[Path] = []
        for d in dirs:
            found = list_images(Path(d))
            if not found:
                raise FileNotFoundError(f"no images under {d}")
            self.paths += found
        self._label = torch.tensor(label, dtype=torch.float32)
        self._is_face = torch.tensor(is_face)
        self.transform = build_transform(resolution, train=train)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.transform(Image.open(self.paths[idx])), self._label, self._is_face


class MixedSmileDataset(Dataset):
    """Concatenation of smile faces (1.0), neutral faces (0.0), and objects (0.0).

    Objects are flagged is_face=False so the training loop can skip the
    discriminator's smile regression loss on them.
    """

    def __init__(self, smile_dirs: list[str | Path],
                 neutral_dirs: list[str | Path],
                 object_dirs: list[str | Path],
                 resolution: int, train: bool = True):
        parts = []
        if smile_dirs:
            parts.append(_LabeledDataset(smile_dirs, 1.0, True, resolution, train))
        if neutral_dirs:
            parts.append(_LabeledDataset(neutral_dirs, 0.0, True, resolution, train))
        if object_dirs:
            parts.append(_LabeledDataset(object_dirs, 0.0, False, resolution, train))
        if not parts:
            raise ValueError("at least one of smile_dirs / neutral_dirs / object_dirs must be non-empty")
        self._data: Dataset = ConcatDataset(parts)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._data[idx]
