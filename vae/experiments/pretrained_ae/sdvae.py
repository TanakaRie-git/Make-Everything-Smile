"""事前学習オートエンコーダ(Stable Diffusion の VAE)を土台に使うためのラッパ。

Diffusion トラックがシャープなのは、笑顔を LoRA で少し足すだけで、画像を描く
decoder は **事前学習済み・凍結の SD-VAE** だから(diffusion/configs/train_lora.yaml
で VAE は frozen)。本実験はその同じ decoder を VAE トラック側でも土台にし、
潜在空間で笑顔方向ベクトルを操作する(スクラッチ VAE-GAN との対照)。

- encode: 画像([-1,1], B×3×H×W) -> 潜在の平均 (B×4×H/8×W/8)
- decode: 潜在 -> 画像([-1,1])
潜在は生のまま(SD の 0.18215 スケールは UNet 用なので autoencode 往復では不要)。
"""

from __future__ import annotations

import torch
from diffusers import AutoencoderKL

DEFAULT_MODEL = "stabilityai/sd-vae-ft-mse"


def load_sdvae(model: str = DEFAULT_MODEL, device: str = "cuda") -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(model).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode(vae: AutoencoderKL, x: torch.Tensor) -> torch.Tensor:
    """画像 -> 潜在(分布の平均=mode を使う。決定論的で編集に向く)。"""
    return vae.encode(x).latent_dist.mean


@torch.no_grad()
def decode(vae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    return vae.decode(z).sample.clamp(-1, 1)
