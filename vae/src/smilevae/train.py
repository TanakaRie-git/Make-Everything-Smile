"""VAE-GAN training (Larsen et al. 2016 style).

Usage:
    python -m smilevae.train configs/smoke64.yaml

The Baseline / +Pareidolia conditions differ only in `data_dirs` (plan §2.3);
everything else comes from the same config schema.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import FlatImageDataset
from .losses import VGGPerceptual
from .models import VAEGAN, reparameterize

DEFAULTS = {
    "resolution": 64,
    "z_dim": 256,
    "latent_size": 0,   # 0 = global vector latent; >=2 = spatial latent map
    "base_channels": 64,
    "max_channels": 512,
    "batch_size": 64,
    "lr": 2e-4,
    "lr_d": 1e-4,         # slower D so it doesn't overpower E+G late in training
    "label_smooth": 0.9,  # real-label target in the D step
    "beta_kl": 1.0,
    "gamma_feat": 1.0,
    "lambda_pixel": 0.5,
    "lambda_perc": 0.0,   # VGG16 perceptual loss weight (0 = off)
    "amp": False,         # mixed precision (needed for 512px on 8GB)
    "steps": 3000,
    "num_workers": 4,
    "log_every": 100,
    "sample_every": 500,
    "ckpt_every": 1000,
    "seed": 0,
}


def load_config(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        cfg.update(yaml.safe_load(f))
    for key in ("data_dirs", "out_dir"):
        if key not in cfg:
            raise KeyError(f"config must define '{key}'")
    return cfg


def d_loss_fn(logits: torch.Tensor, target_value: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target_value))


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"])
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    dataset = FlatImageDataset(cfg["data_dirs"], cfg["resolution"], train=True)
    print(f"dataset: {len(dataset)} images from {cfg['data_dirs']}")
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"],
                   cfg["latent_size"]).to(device)
    opt_eg = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=cfg["lr"], betas=(0.5, 0.999),
    )
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))

    perc = VGGPerceptual().to(device) if cfg["lambda_perc"] > 0 else None
    amp = bool(cfg["amp"]) and device == "cuda"
    scaler_d = torch.amp.GradScaler(enabled=amp)
    scaler_eg = torch.amp.GradScaler(enabled=amp)

    n_fixed = 8 if cfg["resolution"] >= 256 else 16
    fixed = torch.stack([dataset[i] for i in range(0, len(dataset), max(1, len(dataset) // n_fixed))][:n_fixed]).to(device)
    fixed_z = model.sample_prior(n_fixed, device)

    step, t0 = 0, time.time()
    while step < cfg["steps"]:
        for x in loader:
            if step >= cfg["steps"]:
                break
            x = x.to(device, non_blocking=True)

            # --- discriminator: real vs reconstruction vs prior sample ---
            with torch.autocast("cuda", enabled=amp):
                with torch.no_grad():
                    mu, logvar = model.encoder(x)
                    x_rec = model.decoder(reparameterize(mu, logvar))
                    x_pri = model.decoder(model.sample_prior(x.size(0), device))
                d_real, _ = model.discriminator(x)
                d_rec, _ = model.discriminator(x_rec)
                d_pri, _ = model.discriminator(x_pri)
                loss_d = d_loss_fn(d_real, cfg["label_smooth"]) + 0.5 * (d_loss_fn(d_rec, 0.0) + d_loss_fn(d_pri, 0.0))
            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # --- encoder + decoder ---
            with torch.autocast("cuda", enabled=amp):
                mu, logvar = model.encoder(x)
                x_rec = model.decoder(reparameterize(mu, logvar))
                x_pri = model.decoder(model.sample_prior(x.size(0), device))
                with torch.no_grad():
                    _, f_real = model.discriminator(x)
                d_rec, f_rec = model.discriminator(x_rec)
                d_pri, _ = model.discriminator(x_pri)

                loss_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).flatten(1).sum(1).mean() / mu[0].numel()
                loss_feat = F.mse_loss(f_rec, f_real)
                loss_pix = F.l1_loss(x_rec, x)
                loss_perc = perc(x_rec, x) if perc is not None else torch.zeros((), device=device)
                loss_adv = 0.5 * (d_loss_fn(d_rec, 1.0) + d_loss_fn(d_pri, 1.0))
                loss_eg = (
                    cfg["beta_kl"] * loss_kl
                    + cfg["gamma_feat"] * loss_feat
                    + cfg["lambda_pixel"] * loss_pix
                    + cfg["lambda_perc"] * loss_perc
                    + loss_adv
                )
            opt_eg.zero_grad(set_to_none=True)
            scaler_eg.scale(loss_eg).backward()
            scaler_eg.step(opt_eg)
            scaler_eg.update()

            step += 1
            if step % cfg["log_every"] == 0:
                print(
                    f"step {step}/{cfg['steps']} | D {loss_d.item():.3f} | adv {loss_adv.item():.3f} "
                    f"| feat {loss_feat.item():.3f} | pix {loss_pix.item():.3f} | perc {loss_perc.item():.3f} "
                    f"| KL {loss_kl.item():.3f} | {step / (time.time() - t0):.1f} it/s",
                    flush=True,
                )
            if step % cfg["sample_every"] == 0 or step == cfg["steps"]:
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                    rec = model.reconstruct(fixed)
                    pri = model.decoder(fixed_z)
                model.train()
                grid = (torch.cat([fixed, rec, pri]) * 0.5 + 0.5).float()
                save_image(grid, out_dir / "samples" / f"step{step:06d}.png", nrow=n_fixed)
            if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg, "step": step},
                    out_dir / "ckpt" / "last.pt",
                )

    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
