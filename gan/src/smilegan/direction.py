"""Compute the latent smile direction using the GAN generator's encoder.

Mirrors vae/direction.py exactly; the only difference is that G.encode() is
called instead of the VAE's E(x) → (mu, logvar) — the GAN encoder is
deterministic, so there is no reparameterisation step.

The direction lives in the same spatial-latent space as the G bottleneck, so
it can be used in edit.py --mode latent as an alternative to conditional
generation.  This allows a direct, apples-to-apples comparison with the VAE
track's latent directions.

Each --pair gives one (smile_dir, neutral_dir) group; per-group directions are
averaged before normalising (passing male/female pairs separately mitigates
the gender component entangled in CelebA's Smiling attribute).

direction = normalize(mean_over_pairs(mean(z_smile) - mean(z_neutral)))

Also stores proj_std (std of projections onto the direction) so that
edit.py --mode latent can express alpha in comparable units to the VAE.

Usage:
    python -m smilegan.direction --ckpt outputs/smoke64_gan/ckpt/last.pt \
        --pair data/raw/celeba/smile_male data/raw/celeba/neutral_male \
        --pair data/raw/celeba/smile_female data/raw/celeba/neutral_female \
        --out outputs/smoke64_gan/smile_direction.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import build_transform, list_images
from .models import Generator


class _FolderDataset(Dataset):
    def __init__(self, folder: str, resolution: int):
        self.paths = list_images(Path(folder))
        if not self.paths:
            raise FileNotFoundError(f"no images under {folder}")
        self.transform = build_transform(resolution, train=False)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]))


@torch.no_grad()
def encode_folder(G: Generator, folder: str, resolution: int,
                  device: str, batch_size: int = 128) -> torch.Tensor:
    """Returns (N, z_channels, latent_size, latent_size) tensor of encoded latents."""
    ds = _FolderDataset(folder, resolution)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    zs = []
    for x in loader:
        zs.append(G.encode(x.to(device)).cpu())
    return torch.cat(zs)


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
    G = Generator(
        cfg["resolution"], cfg["z_channels"], cfg["latent_size"],
        cfg["base_channels"], cfg["max_channels"], cfg["n_res"], cfg["style_dim"],
    ).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    directions, all_zs, n_total = [], [], []
    for smile_dir, neutral_dir in args.pair:
        z_s = encode_folder(G, smile_dir, cfg["resolution"], device)
        z_n = encode_folder(G, neutral_dir, cfg["resolution"], device)
        gap = z_s.mean(0) - z_n.mean(0)  # (z_channels, latent_size, latent_size)
        print(f"pair {Path(smile_dir).name} vs {Path(neutral_dir).name}: "
              f"|gap|={gap.norm():.3f}, n={len(z_s)}+{len(z_n)}")
        directions.append(gap)
        all_zs += [z_s.flatten(1), z_n.flatten(1)]
        n_total.append(len(z_s) + len(z_n))

    direction = torch.stack(directions).mean(0)
    direction = direction / direction.norm()
    proj_std = torch.cat(all_zs).matmul(direction.flatten()).std().item()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"direction": direction, "proj_std": proj_std,
         "pairs": args.pair, "n_total": sum(n_total)},
        args.out,
    )
    print(f"saved {args.out}: shape={tuple(direction.shape)}, proj_std={proj_std:.3f}")


if __name__ == "__main__":
    main()
