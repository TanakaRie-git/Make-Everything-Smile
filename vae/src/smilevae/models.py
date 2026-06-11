"""VAE-GAN (Larsen et al. 2016) building blocks.

Resolution-flexible: the number of stride-2 stages is derived from the input
resolution (down to a 4x4 feature map). The discriminator exposes an
intermediate feature map used for the feature-wise reconstruction loss.
"""

import math

import torch
import torch.nn as nn


def _num_stages(resolution: int) -> int:
    n = int(math.log2(resolution / 4))
    if 4 * 2**n != resolution:
        raise ValueError(f"resolution must be 4*2^n, got {resolution}")
    return n


def _channels(stage: int, base: int, max_ch: int) -> int:
    return min(base * 2**stage, max_ch)


class Encoder(nn.Module):
    def __init__(self, resolution: int = 64, z_dim: int = 256, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution)
        layers = []
        in_ch = 3
        for s in range(stages):
            out_ch = _channels(s, base, max_ch)
            layers += [
                nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_ch * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(in_ch * 4 * 4, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


class Decoder(nn.Module):
    def __init__(self, resolution: int = 64, z_dim: int = 256, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution)
        top_ch = _channels(stages - 1, base, max_ch)
        self.top_ch = top_ch
        self.fc = nn.Linear(z_dim, top_ch * 4 * 4)
        layers = []
        in_ch = top_ch
        for s in reversed(range(stages)):
            out_ch = _channels(s - 1, base, max_ch) if s > 0 else base
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        layers += [nn.Conv2d(in_ch, 3, 3, padding=1), nn.Tanh()]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.top_ch, 4, 4)
        return self.conv(h)


class Discriminator(nn.Module):
    """DCGAN-style discriminator; `forward` returns (logit, feature_map).

    The feature map is taken after the penultimate conv stage and is the
    target space for the VAE-GAN feature-wise reconstruction loss.
    """

    def __init__(self, resolution: int = 64, base: int = 64, max_ch: int = 512):
        super().__init__()
        stages = _num_stages(resolution)
        blocks = []
        in_ch = 3
        for s in range(stages):
            out_ch = _channels(s, base, max_ch)
            block = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)]
            if s > 0:  # no norm on the first block, per DCGAN
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            blocks.append(nn.Sequential(*block))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(in_ch, 1, 4)  # 4x4 -> 1x1 logit

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        feat = None
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == len(self.blocks) - 2:
                feat = h
        return self.head(h).flatten(1), feat


class VAEGAN(nn.Module):
    def __init__(self, resolution: int = 64, z_dim: int = 256, base: int = 64, max_ch: int = 512):
        super().__init__()
        self.resolution = resolution
        self.z_dim = z_dim
        self.encoder = Encoder(resolution, z_dim, base, max_ch)
        self.decoder = Decoder(resolution, z_dim, base, max_ch)
        self.discriminator = Discriminator(resolution, base, max_ch)

    def reconstruct(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar) if sample else mu
        return self.decoder(z)
