"""Apply the smile direction to object images at weak/mid/strong strengths.

For each input image: z = mu(E(x)); output D(z + alpha * proj_std * direction).
Saves a labeled contact sheet (rows = inputs; columns = input | recon |
weak | mid | strong), upscaled so low-resolution runs stay readable.

Usage:
    python -m smilevae.edit --ckpt outputs/smoke64/ckpt/last.pt \
        --direction outputs/smoke64/smile_direction.pt \
        --input-dir ../data/raw/objects_smoke --out-dir outputs/smoke64/edits \
        --strengths 1 2 3
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import FlatImageDataset
from .models import VAEGAN

STRENGTH_NAMES = ["weak", "mid", "strong"]
HEADER_H = 24
LABEL_W = 110
PAD = 4


def to_pil(t: torch.Tensor, upscale: int) -> Image.Image:
    arr = ((t.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    return img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)


def make_sheet(rows: list[tuple[str, list[Image.Image]]], col_names: list[str]) -> Image.Image:
    tile = rows[0][1][0].width
    n_cols, n_rows = len(col_names), len(rows)
    sheet = Image.new(
        "RGB",
        (LABEL_W + n_cols * (tile + PAD), HEADER_H + n_rows * (tile + PAD)),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(col_names):
        draw.text((LABEL_W + c * (tile + PAD) + tile // 2, HEADER_H // 2),
                  name, fill=(255, 255, 255), anchor="mm")
    for r, (label, tiles) in enumerate(rows):
        y = HEADER_H + r * (tile + PAD)
        draw.text((4, y + tile // 2), label[:16], fill=(200, 200, 200), anchor="lm")
        for c, im in enumerate(tiles):
            sheet.paste(im, (LABEL_W + c * (tile + PAD), y))
    return sheet


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strengths", type=float, nargs=3, default=[1.0, 2.0, 3.0],
                        help="weak/mid/strong, in units of latent proj_std")
    parser.add_argument("--limit", type=int, default=None, help="max input images")
    parser.add_argument("--upscale", type=int, default=3, help="nearest-neighbor tile upscale")
    parser.add_argument("--name", default="contact_sheet", help="output file stem")
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

    col_names = ["input", "recon"] + [
        f"{nm} (α={a:g})" for nm, a in zip(STRENGTH_NAMES, args.strengths)
    ]
    rows = []
    for i in range(n):
        x = ds[i].unsqueeze(0).to(device)
        mu, _ = model.encoder(x)
        outputs = [x[0], model.decoder(mu)[0]]
        for alpha in args.strengths:
            outputs.append(model.decoder(mu + alpha * scale * direction)[0])
        rows.append((ds.paths[i].stem, [to_pil(t, args.upscale) for t in outputs]))

    sheet = make_sheet(rows, col_names)
    out_path = out_dir / f"{args.name}.png"
    sheet.save(out_path)
    print(f"saved {n} rows -> {out_path} (columns: {' | '.join(col_names)})")


if __name__ == "__main__":
    main()
