"""事前学習 SD-VAE の潜在空間で笑顔方向ベクトルを算出する。

smilevae.direction と同じ設計(男女別ペアの平均で性別もつれを軽減):
    direction = normalize(mean_over_pairs(mean(z_smile) - mean(z_neutral)))
方向への射影の std も保存し、編集強度を std 単位(α)で指定できるようにする。

使い方(cd vae から):
    uv run python experiments/pretrained_ae/build_direction.py \
        --pair data/raw/celeba_hq/smile_male   data/raw/celeba_hq/neutral_male \
        --pair data/raw/celeba_hq/smile_female data/raw/celeba_hq/neutral_female \
        --out outputs/pretrained_ae/smile_direction.pt --resolution 512
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smilevae.data import FlatImageDataset
from sdvae import load_sdvae, encode


@torch.no_grad()
def encode_folder(vae, folder: str, resolution: int, device: str,
                  batch_size: int, limit: int | None) -> torch.Tensor:
    ds = FlatImageDataset([folder], resolution, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4)
    zs, seen = [], 0
    for x in loader:
        zs.append(encode(vae, x.to(device)).cpu())
        seen += x.size(0)
        if limit and seen >= limit:
            break
    return torch.cat(zs)[:limit] if limit else torch.cat(zs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("SMILE_DIR", "NEUTRAL_DIR"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-per-folder", type=int, default=None,
                        help="各フォルダの使用枚数上限(高速化用)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = load_sdvae(args.model, device)

    directions, all_zs, n_total = [], [], 0
    for smile_dir, neutral_dir in args.pair:
        z_s = encode_folder(vae, smile_dir, args.resolution, device, args.batch_size, args.limit_per_folder)
        z_n = encode_folder(vae, neutral_dir, args.resolution, device, args.batch_size, args.limit_per_folder)
        gap = z_s.mean(0) - z_n.mean(0)
        print(f"pair {Path(smile_dir).name} vs {Path(neutral_dir).name}: "
              f"|gap|={gap.norm():.3f}, n={len(z_s)}+{len(z_n)}")
        directions.append(gap)
        all_zs += [z_s, z_n]
        n_total += len(z_s) + len(z_n)

    direction = torch.stack(directions).mean(0)
    direction = direction / direction.norm()
    proj_std = torch.cat(all_zs).flatten(1).matmul(direction.flatten()).std().item()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"direction": direction, "proj_std": proj_std,
                "pairs": args.pair, "n_total": n_total,
                "model": args.model, "resolution": args.resolution}, args.out)
    print(f"saved {args.out}: shape={tuple(direction.shape)}, proj_std={proj_std:.3f}")


if __name__ == "__main__":
    main()
