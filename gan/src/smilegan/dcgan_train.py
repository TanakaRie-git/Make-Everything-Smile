"""DCGAN smile transfer training.

Train G: neutral_face → smile_face on CelebA-HQ.
No conditioning scalar, no cycle loss, no smile regression — plain DCGAN.

D step:
  real_smile images → real
  G(neutral) images → fake

G step:
  adversarial: D(G(neutral)) should look real
  reconstruction: G(smile) ≈ smile  (identity; prevents mode collapse)

Usage:
  uv run python -m smilegan.dcgan_train configs/dcgan256.yaml
"""

import argparse
import copy
import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .data import FlatImageDataset
from .dcgan_models import Discriminator, Generator

DEFAULTS: dict = {
    "resolution": 256,
    "base_channels": 64,
    "max_channels": 512,
    "batch_size": 8,
    "lr_g": 2.0e-4,
    "lr_d": 1.0e-4,
    "lambda_rec": 10.0,
    "amp": False,
    "steps": 60000,
    "num_workers": 4,
    "log_every": 50,
    "sample_every": 1000,
    "ckpt_every": 2000,
    "seed": 0,
}


def load_config(path: str) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        cfg.update(yaml.safe_load(f))
    for key in ("smile_dirs", "neutral_dirs", "out_dir"):
        if key not in cfg:
            raise KeyError(f"config must define '{key}'")
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

    ds_smile = FlatImageDataset(cfg["smile_dirs"], cfg["resolution"], train=True)
    ds_neutral = FlatImageDataset(cfg["neutral_dirs"], cfg["resolution"], train=True)
    print(f"smile: {len(ds_smile)}  neutral: {len(ds_neutral)}")

    kw = dict(batch_size=cfg["batch_size"], shuffle=True,
              num_workers=cfg["num_workers"], drop_last=True,
              persistent_workers=cfg["num_workers"] > 0)
    smile_iter = cycle(DataLoader(ds_smile, **kw))
    neutral_iter = cycle(DataLoader(ds_neutral, **kw))

    G = Generator(cfg["resolution"], cfg["base_channels"], cfg["max_channels"]).to(device)
    D = Discriminator(cfg["resolution"], cfg["base_channels"], cfg["max_channels"]).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))

    amp = bool(cfg["amp"]) and device == "cuda"
    scaler_d = torch.amp.GradScaler(enabled=amp)
    scaler_g = torch.amp.GradScaler(enabled=amp)

    # Fixed samples for visualisation
    rng = torch.Generator()
    rng.manual_seed(cfg["seed"])
    n_fixed = 8
    fixed_neutral = torch.stack(
        [ds_neutral[i] for i in torch.randperm(len(ds_neutral), generator=rng)[:n_fixed]]
    ).to(device)
    fixed_smile = torch.stack(
        [ds_smile[i] for i in torch.randperm(len(ds_smile), generator=rng)[:n_fixed]]
    ).to(device)

    step, t0 = 0, time.time()
    while step < cfg["steps"]:
        real_smile = next(smile_iter).to(device, non_blocking=True)
        neutral = next(neutral_iter).to(device, non_blocking=True)

        # ── Discriminator ─────────────────────────────────────────────────────
        with torch.autocast("cuda", enabled=amp):
            with torch.no_grad():
                fake_smile = G(neutral)
            loss_d = adv_loss(D(real_smile), 0.9) + adv_loss(D(fake_smile), 0.0)

        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(loss_d).backward()
        scaler_d.step(opt_d)
        scaler_d.update()

        # ── Generator ─────────────────────────────────────────────────────────
        with torch.autocast("cuda", enabled=amp):
            fake_smile = G(neutral)
            loss_g_adv = adv_loss(D(fake_smile), 1.0)
            # Identity mapping on smile images: G(smile) ≈ smile
            loss_g_rec = F.l1_loss(G(real_smile), real_smile)
            loss_g = loss_g_adv + cfg["lambda_rec"] * loss_g_rec

        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(loss_g).backward()
        scaler_g.step(opt_g)
        scaler_g.update()

        step += 1

        if step % cfg["log_every"] == 0:
            print(
                f"step {step}/{cfg['steps']} | "
                f"D {loss_d.item():.3f} | "
                f"G adv {loss_g_adv.item():.3f}  rec {loss_g_rec.item():.3f} | "
                f"{step / (time.time() - t0):.1f} it/s",
                flush=True,
            )

        if step % cfg["sample_every"] == 0 or step == cfg["steps"]:
            G.eval()
            with torch.no_grad(), torch.autocast("cuda", enabled=amp):
                fixed_fake = G(fixed_neutral)
                fixed_rec = G(fixed_smile)
            G.train()
            # rows: neutral | G(neutral) | smile | G(smile)
            grid = torch.cat([fixed_neutral, fixed_fake, fixed_smile, fixed_rec])
            grid = (grid * 0.5 + 0.5).float().clamp(0, 1)
            save_image(grid, out_dir / "samples" / f"step{step:06d}.png",
                       nrow=n_fixed)

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
