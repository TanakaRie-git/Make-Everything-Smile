"""DCGAN-style generator and discriminator for smile transfer.

G(x) → y: neutral image → smile image. No conditioning scalar.
D(y) → logit: real smile vs generated smile. No smile regression head.

Contrast with models.py (StarGAN):
  - No AdaIN / SmileMLP conditioning
  - No smile regression head on D
  - Simpler training signal: adversarial + reconstruction only
"""

import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _n_down(resolution: int, bottom: int = 4) -> int:
    n = int(math.log2(resolution / bottom))
    if bottom * 2**n != resolution:
        raise ValueError(f"resolution must be {bottom}*2^n, got {resolution}")
    return n


# ── Generator ────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """Encoder-decoder: G(x) → y with no smile conditioning.

    Encoder: strided Conv → BN → LeakyReLU (×n_down stages, 256→16 at 256px).
    Decoder: Upsample → Conv → BN → ReLU  (×n_down stages, 16→256).
    """

    def __init__(self, resolution: int = 256, base: int = 64, max_ch: int = 512):
        super().__init__()
        n = _n_down(resolution, bottom=16)  # 256px → 4 stages → 16×16 bottleneck

        # channel schedule: [64, 128, 256, 512] for base=64, max_ch=512
        chs = [min(base * 2**i, max_ch) for i in range(n)]

        enc = []
        in_ch = 3
        for i, out_ch in enumerate(chs):
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
            if i > 0:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            enc.append(nn.Sequential(*layers))
            in_ch = out_ch
        self.encoder = nn.ModuleList(enc)

        dec = []
        for out_ch in reversed(chs[:-1]):  # 512→256→128→64
            dec.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        dec.append(nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_ch, 3, 3, padding=1),
            nn.Tanh(),
        ))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for enc in self.encoder:
            h = enc(h)
        return self.decoder(h)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """DCGAN PatchGAN discriminator with spectral norm.

    No smile regression head — plain real/fake adversarial only.
    Goes from resolution down to 4×4 then a 4×4 conv → scalar logit.
    """

    def __init__(self, resolution: int = 256, base: int = 64, max_ch: int = 512):
        super().__init__()
        n = _n_down(resolution, bottom=4)  # 256px → 6 stages → 4×4

        layers: list[nn.Module] = []
        in_ch = 3
        for i in range(n):
            out_ch = min(base * 2**i, max_ch)
            conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
            layers.append(conv)
            if i > 0:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
        layers.append(spectral_norm(nn.Conv2d(in_ch, 1, 4)))  # 4×4 → 1×1
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)
