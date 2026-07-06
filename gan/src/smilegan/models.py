"""StarGAN-style generator and multi-task discriminator for smile editing.

Generator G(x, s) — x: RGB image (3×H×W), s: smile strength scalar ∈ ℝ.
  Encoder    → spatial feature map (z_channels × latent_size × latent_size)
  Bottleneck → n_res AdaIN-conditioned residual blocks (smile injected here)
  Decoder    → RGB image (3×H×W) via nearest-neighbour upsampling

Discriminator D(x):
  Shared DCGAN backbone with spectral-normalised convolutions for stability.
  adv_head   → PatchGAN real/fake logit (collapsed to scalar at 4×4 bottom)
  smile_head → smile regression score from global average pooling
               (trained only on labeled face images — see train.py)

The primary edit path is conditional: G(x, s) maps any image to a version
with smile strength s.  The encoder is also exposed separately so direction.py
can compute a post-hoc smile direction in latent space for cross-track
comparison with the VAE.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _num_stages(resolution: int, bottom: int = 4) -> int:
    n = int(math.log2(resolution / bottom))
    if bottom * 2**n != resolution:
        raise ValueError(f"resolution must be {bottom}*2^n, got {resolution}")
    return n


def _channels(stage: int, base: int, max_ch: int) -> int:
    return min(base * 2**stage, max_ch)


# ── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """RGB → (z_channels × latent_size × latent_size) spatial feature map.

    Deterministic (no reparameterisation); uses InstanceNorm which is standard
    for image-to-image translation and avoids batch-size sensitivity.
    """

    def __init__(self, resolution: int = 64, z_channels: int = 64,
                 latent_size: int = 8, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution, latent_size)
        layers, in_ch = [], 3
        for s in range(stages):
            out_ch = _channels(s, base, max_ch)
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, z_channels, 1)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── AdaIN conditioning ────────────────────────────────────────────────────────

class SmileMLP(nn.Module):
    """Scalar smile strength s → style_dim vector for AdaIN conditioning.

    Three-layer MLP; the final layer is zero-initialised so the network starts
    as a near-identity (smile perturbations grow from zero at the start of
    training, keeping the generator stable).
    """

    def __init__(self, style_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
            nn.ReLU(inplace=True),
            nn.Linear(style_dim, style_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s.view(-1, 1).float())


class AdaIN(nn.Module):
    """Adaptive Instance Normalization: IN then affine-shift by style."""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.linear = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.linear(style).chunk(2, dim=1)
        return h * (gamma[:, :, None, None] + 1.0) + beta[:, :, None, None]


class CondResBlock(nn.Module):
    """Residual block with AdaIN smile conditioning."""

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.adain1 = AdaIN(channels, style_dim)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.adain2 = AdaIN(channels, style_dim)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.adain1(x, style), inplace=True)
        h = self.conv1(h)
        h = F.relu(self.adain2(h, style), inplace=True)
        h = self.conv2(h)
        return x + h


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    def __init__(self, resolution: int = 64, z_channels: int = 64,
                 latent_size: int = 8, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution, latent_size)
        top_ch = _channels(stages - 1, base, max_ch)
        layers = [
            nn.Conv2d(z_channels, top_ch, 3, padding=1),
            nn.InstanceNorm2d(top_ch, affine=True),
            nn.ReLU(inplace=True),
        ]
        in_ch = top_ch
        for s in reversed(range(stages)):
            out_ch = _channels(s - 1, base, max_ch) if s > 0 else base
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 3, 3, padding=1), nn.Tanh()]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.conv(z)


# ── Generator ────────────────────────────────────────────────────────────────

class Generator(nn.Module):
    """G(x, s) → y: inject smile strength s into the spatial latent bottleneck.

    s=0 → reconstruction, s=1 → smiling, s>1 → stronger smile (extrapolation).
    encode() and decode() are exposed separately so direction.py can reuse the
    encoder for post-hoc smile-direction analysis (analogous to the VAE track),
    and decode_raw() skips the AdaIN blocks for pure latent-perturbation edits.
    """

    def __init__(self, resolution: int = 64, z_channels: int = 64,
                 latent_size: int = 8, base: int = 64, max_ch: int = 512,
                 n_res: int = 4, style_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(resolution, z_channels, latent_size, base, max_ch)
        self.smile_mlp = SmileMLP(style_dim)
        self.res_blocks = nn.ModuleList(
            [CondResBlock(z_channels, style_dim) for _ in range(n_res)]
        )
        self.decoder = Decoder(resolution, z_channels, latent_size, base, max_ch)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, smile: torch.Tensor) -> torch.Tensor:
        style = self.smile_mlp(smile)
        for block in self.res_blocks:
            z = block(z, style)
        return self.decoder(z)

    def decode_raw(self, z: torch.Tensor) -> torch.Tensor:
        """Bypass AdaIN blocks; decode z directly. Used in latent-direction mode."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor, smile: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), smile)


# ── Discriminator ─────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """Multi-task PatchGAN: adversarial real/fake + smile regression.

    spectral_norm on every Conv2d keeps the Lipschitz constant bounded, which
    stabilises adversarial training without needing a gradient penalty.

    smile_head is trained only on labeled face images; object images contribute
    only to the adversarial head (caller's responsibility — see train.py).
    """

    def __init__(self, resolution: int = 64, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution)
        blocks, in_ch = [], 3
        for s in range(stages):
            out_ch = _channels(s, base, max_ch)
            conv = spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1))
            blocks.append(nn.Sequential(conv, nn.LeakyReLU(0.2, inplace=True)))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.adv_head = spectral_norm(nn.Conv2d(in_ch, 1, 4))
        self.smile_head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (adv_logit, smile_score, penultimate_feat)."""
        h = x
        feat = None
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == len(self.blocks) - 2:
                feat = h
        adv = self.adv_head(h).flatten(1)
        smile_score = self.smile_head(h.mean(dim=(2, 3))).squeeze(1)
        return adv, smile_score, feat
