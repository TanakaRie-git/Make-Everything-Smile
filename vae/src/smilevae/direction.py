"""Compute the latent smile direction from smiling vs non-smiling face folders.

direction = normalize(mean(mu_smile) - mean(mu_neutral))

Also stores the std of latent projections onto the direction, so edit strengths
(weak/mid/strong) can be specified in units of that std (plan §0.3).

Usage:
    python -m smilevae.direction --ckpt outputs/smoke64/ckpt/last.pt \
        --smile-dir ../data/processed/celeba_smile --neutral-dir ../data/processed/celeba_neutral \
        --out outputs/smoke64/smile_direction.pt
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import FlatImageDataset
from .models import VAEGAN


@torch.no_grad()
def encode_folder(model: VAEGAN, folder: str, resolution: int, device: str, batch_size: int = 128) -> torch.Tensor:
    ds = FlatImageDataset([folder], resolution, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    mus = []
    for x in loader:
        mu, _ = model.encoder(x.to(device))
        mus.append(mu.cpu())
    return torch.cat(mus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--smile-dir", required=True)
    parser.add_argument("--neutral-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    mu_s = encode_folder(model, args.smile_dir, cfg["resolution"], device)
    mu_n = encode_folder(model, args.neutral_dir, cfg["resolution"], device)
    direction = mu_s.mean(0) - mu_n.mean(0)
    gap = direction.norm().item()
    direction = direction / direction.norm()
    proj_std = torch.cat([mu_s, mu_n]).matmul(direction).std().item()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"direction": direction, "proj_std": proj_std, "mean_gap": gap,
         "n_smile": len(mu_s), "n_neutral": len(mu_n)},
        args.out,
    )
    print(f"saved {args.out}: |mu_s-mu_n|={gap:.3f}, proj_std={proj_std:.3f}, "
          f"n={len(mu_s)}+{len(mu_n)}")


if __name__ == "__main__":
    main()
