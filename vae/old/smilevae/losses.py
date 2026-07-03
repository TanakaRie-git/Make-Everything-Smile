"""Auxiliary losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptual(nn.Module):
    """L1 distance in VGG16 relu3_3 feature space. Inputs in [-1, 1].

    Images larger than `max_side` are downsampled first — at 512px the VGG
    activations would otherwise dominate the memory budget.
    """

    def __init__(self, max_side: int = 256):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg
        self.max_side = max_side
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_side:
            x = F.interpolate(x, size=self.max_side, mode="bilinear", align_corners=False)
        x = ((x + 1) / 2 - self.mean) / self.std
        return self.vgg(x)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._features(x), self._features(target))
