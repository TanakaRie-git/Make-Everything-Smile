"""Compute the latent smile direction from smiling vs non-smiling face folders.

Each --pair gives one (smile_dir, neutral_dir) group; the per-group directions
are averaged before normalizing. Passing male and female groups separately
cancels the gender component that CelebA's Smiling attribute is entangled with.

direction = normalize(mean_over_pairs(mean(mu_smile) - mean(mu_neutral)))

Also stores the std of latent projections onto the direction, so edit strengths
(weak/mid/strong) can be specified in units of that std (plan §0.3).

Usage:
    python -m smilevae.direction --ckpt outputs/smoke64/ckpt/last.pt \
        --pair data/raw/celeba/smile_male data/raw/celeba/neutral_male \
        --pair data/raw/celeba/smile_female data/raw/celeba/neutral_female \
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
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("SMILE_DIR", "NEUTRAL_DIR"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"],
                   cfg.get("latent_size", 0)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    directions, all_mus, n_total = [], [], []
    for smile_dir, neutral_dir in args.pair:
        mu_s = encode_folder(model, smile_dir, cfg["resolution"], device)
        mu_n = encode_folder(model, neutral_dir, cfg["resolution"], device)
        gap = mu_s.mean(0) - mu_n.mean(0)
        print(f"pair {Path(smile_dir).name} vs {Path(neutral_dir).name}: "
              f"|gap|={gap.norm():.3f}, n={len(mu_s)}+{len(mu_n)}")
        directions.append(gap)
        all_mus += [mu_s, mu_n]
        n_total.append(len(mu_s) + len(mu_n))

    direction = torch.stack(directions).mean(0)
    direction = direction / direction.norm()
    proj_std = torch.cat(all_mus).flatten(1).matmul(direction.flatten()).std().item()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"direction": direction, "proj_std": proj_std,
         "pairs": args.pair, "n_total": sum(n_total)},
        args.out,
    )
    print(f"saved {args.out}: shape={tuple(direction.shape)}, proj_std={proj_std:.3f}")


if __name__ == "__main__":
    main()
