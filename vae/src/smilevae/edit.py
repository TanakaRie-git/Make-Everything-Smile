"""Apply the smile direction to object images at weak/mid/strong strengths.

For each input image: z = mu(E(x)); output D(z + alpha * proj_std * direction).
Saves per-image rows (input | recon | weak | mid | strong) and one contact sheet.

Usage:
    python -m smilevae.edit --ckpt outputs/smoke64/ckpt/last.pt \
        --direction outputs/smoke64/smile_direction.pt \
        --input-dir ../data/processed/objects_smoke --out-dir outputs/smoke64/edits \
        --strengths 1.5 3 5
"""

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .data import FlatImageDataset
from .models import VAEGAN

STRENGTH_NAMES = ["weak", "mid", "strong"]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strengths", type=float, nargs=3, default=[1.5, 3.0, 5.0],
                        help="weak/mid/strong, in units of latent proj_std")
    parser.add_argument("--limit", type=int, default=None, help="max input images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dir_state = torch.load(args.direction, map_location=device, weights_only=False)
    direction = dir_state["direction"].to(device)
    scale = dir_state["proj_std"]

    ds = FlatImageDataset([args.input_dir], cfg["resolution"], train=False)
    n = min(len(ds), args.limit) if args.limit else len(ds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        x = ds[i].unsqueeze(0).to(device)
        mu, _ = model.encoder(x)
        outputs = [x, model.decoder(mu)]
        for alpha in args.strengths:
            outputs.append(model.decoder(mu + alpha * scale * direction))
        row = torch.cat(outputs) * 0.5 + 0.5
        save_image(row, out_dir / f"{ds.paths[i].stem}_edit.png", nrow=len(outputs))
        rows.append(row.cpu())

    sheet = torch.cat(rows)
    save_image(sheet, out_dir / "_contact_sheet.png", nrow=2 + len(args.strengths))
    print(f"saved {n} rows -> {out_dir} (columns: input | recon | "
          f"{' | '.join(f'{nm}={a}' for nm, a in zip(STRENGTH_NAMES, args.strengths))})")


if __name__ == "__main__":
    main()
