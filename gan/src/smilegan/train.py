"""StarGAN v1-style training for smile attribute editing.

Usage:
    python -m smilegan.train configs/smoke64.yaml

One D step + one G step per iteration.

D step:
  1. Real images (faces + objects): adversarial real loss (label smoothing).
  2. Smile regression on real FACE images only (MSE vs ground-truth label).
     Object images have no smile label so they are excluded from this term.
  3. Fake images from G: adversarial fake loss.

G step:
  1. Adversarial: generated images should be classified as real by D.
  2. Classification: D_smile(G(face, target)) ≈ target  [faces only].
  3. Reconstruction: G(x, source_label) ≈ x  (pixel L1; enforces s=source → identity).
  4. Cycle consistency: G(G(face, target), source) ≈ face  [faces only; optional].
  5. Perceptual: VGG feature L1 on the reconstruction  [optional].

The Baseline / +Pareidolia conditions differ only in object_dirs (plan §2.3).
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

from .data import MixedSmileDataset
from .losses import VGGPerceptual
from .models import Discriminator, Generator

DEFAULTS: dict = {
    "resolution": 64,
    "z_channels": 64,
    "latent_size": 8,
    "base_channels": 96,
    "max_channels": 512,
    "n_res": 4,
    "style_dim": 64,
    "batch_size": 32,
    "lr_g": 2e-4,
    "lr_d": 1e-4,
    "label_smooth": 0.9,
    "lambda_cls": 1.0,
    "lambda_rec": 10.0,
    "lambda_cyc": 10.0,
    "lambda_perc": 0.0,
    "amp": False,
    "init_from": None,
    "steps": 20000,
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
    for key in ("smile_dirs", "neutral_dirs", "out_dir"):
        if key not in cfg:
            raise KeyError(f"config must define '{key}'")
    cfg.setdefault("object_dirs", [])
    return cfg


def adv_loss(logits: torch.Tensor, target: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, target))


def train(cfg: dict) -> None:
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["out_dir"])
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpt").mkdir(exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    dataset = MixedSmileDataset(
        smile_dirs=cfg["smile_dirs"],
        neutral_dirs=cfg["neutral_dirs"],
        object_dirs=cfg["object_dirs"],
        resolution=cfg["resolution"],
        train=True,
    )
    print(f"dataset: {len(dataset)} images")
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    G = Generator(
        cfg["resolution"], cfg["z_channels"], cfg["latent_size"],
        cfg["base_channels"], cfg["max_channels"], cfg["n_res"], cfg["style_dim"],
    ).to(device)
    D = Discriminator(
        cfg["resolution"], cfg["base_channels"], cfg["max_channels"],
    ).to(device)

    if cfg["init_from"]:
        ckpt = torch.load(cfg["init_from"], map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        print(f"warm-started from {cfg['init_from']} (step {ckpt.get('step')})")

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    perc = VGGPerceptual().to(device) if cfg["lambda_perc"] > 0 else None
    amp = bool(cfg["amp"]) and device == "cuda"
    scaler_d = torch.amp.GradScaler(enabled=amp)
    scaler_g = torch.amp.GradScaler(enabled=amp)

    # Fixed samples for visualisation (seeded to be reproducible)
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n_fixed = min(8, len(dataset))
    fixed_idx = torch.randperm(len(dataset), generator=rng)[:n_fixed].tolist()
    fixed_items = [dataset[i] for i in fixed_idx]
    fixed_x = torch.stack([it[0] for it in fixed_items]).to(device)
    fixed_label = torch.stack([it[1] for it in fixed_items]).to(device)
    fixed_target = (1.0 - fixed_label).clamp(0.0, 1.0)

    step, t0 = 0, time.time()
    while step < cfg["steps"]:
        for x, label, is_face in loader:
            if step >= cfg["steps"]:
                break
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            is_face = is_face.to(device, non_blocking=True)
            face_mask = is_face.bool()
            # Target label: flip for faces (neutral→smile, smile→neutral),
            # always 1.0 for objects (make them smile).
            target_label = torch.where(face_mask, 1.0 - label, torch.ones_like(label))

            # ── Discriminator ─────────────────────────────────────────────────
            with torch.autocast("cuda", enabled=amp):
                with torch.no_grad():
                    x_fake = G(x, target_label)
                adv_real, smile_real, _ = D(x)
                adv_fake, _, _ = D(x_fake)
                loss_d_real = adv_loss(adv_real, cfg["label_smooth"])
                loss_d_fake = adv_loss(adv_fake, 0.0)
                # Smile regression only on labeled face images
                if face_mask.any():
                    loss_d_cls = F.mse_loss(smile_real[face_mask], label[face_mask])
                else:
                    loss_d_cls = x.new_zeros(())
                loss_d = loss_d_real + loss_d_fake + cfg["lambda_cls"] * loss_d_cls

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ── Generator ─────────────────────────────────────────────────────
            with torch.autocast("cuda", enabled=amp):
                x_fake = G(x, target_label)
                adv_fake2, smile_fake, _ = D(x_fake)
                x_rec = G(x, label)
                loss_g_adv = adv_loss(adv_fake2, 1.0)
                if face_mask.any():
                    loss_g_cls = F.mse_loss(smile_fake[face_mask], target_label[face_mask])
                else:
                    loss_g_cls = x.new_zeros(())
                loss_g_rec = F.l1_loss(x_rec, x)
                # Cycle consistency on face images: G(G(x,target), source) ≈ x
                loss_g_cyc = x.new_zeros(())
                if cfg["lambda_cyc"] > 0 and face_mask.any():
                    x_cyc = G(x_fake[face_mask], label[face_mask])
                    loss_g_cyc = F.l1_loss(x_cyc, x[face_mask])
                loss_g_perc = (
                    perc(x_rec, x) if perc is not None else x.new_zeros(())
                )
                loss_g = (
                    loss_g_adv
                    + cfg["lambda_cls"] * loss_g_cls
                    + cfg["lambda_rec"] * loss_g_rec
                    + cfg["lambda_cyc"] * loss_g_cyc
                    + cfg["lambda_perc"] * loss_g_perc
                )

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            step += 1
            if step % cfg["log_every"] == 0:
                print(
                    f"step {step}/{cfg['steps']} | "
                    f"D {loss_d.item():.3f} (cls {loss_d_cls.item():.3f}) | "
                    f"G adv {loss_g_adv.item():.3f} cls {loss_g_cls.item():.3f} "
                    f"rec {loss_g_rec.item():.3f} cyc {loss_g_cyc.item():.3f} "
                    f"perc {loss_g_perc.item():.3f} | "
                    f"{step / (time.time() - t0):.1f} it/s",
                    flush=True,
                )
            if step % cfg["sample_every"] == 0 or step == cfg["steps"]:
                G.eval()
                with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                    fixed_rec = G(fixed_x, fixed_label)
                    fixed_fake = G(fixed_x, fixed_target)
                G.train()
                grid = torch.cat([fixed_x, fixed_rec, fixed_fake])
                grid = (grid * 0.5 + 0.5).float().clamp(0, 1)
                save_image(grid, out_dir / "samples" / f"step{step:06d}.png",
                           nrow=len(fixed_x))
            if step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
                torch.save(
                    {"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg, "step": step},
                    out_dir / "ckpt" / "last.pt",
                )

    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    train(load_config(parser.parse_args().config))


if __name__ == "__main__":
    main()
