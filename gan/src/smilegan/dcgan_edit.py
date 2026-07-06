"""Apply trained DCGAN to images and produce a contact sheet.

Contact sheet layout (rows = images, columns = input | G(input)):
  input   : original image
  output  : G(input) — smile-transferred result

Usage:
  # Pareidolia verification:
  uv run python -m smilegan.dcgan_edit \
      --ckpt outputs/dcgan256/ckpt/last.pt \
      --input-dir data/raw/faces_in_things/crops/neutral \
      --out-dir outputs/dcgan256/edits \
      --limit 20

  # Compare against real face inputs:
  uv run python -m smilegan.dcgan_edit \
      --ckpt outputs/dcgan256/ckpt/last.pt \
      --input-dir data/raw/celeba_hq/neutral_female \
      --out-dir outputs/dcgan256/edits_face \
      --name celeba_sheet
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import FlatImageDataset
from .dcgan_models import Generator

HEADER_H = 24
LABEL_W = 110
PAD = 4
COL_NAMES = ["input", "G(input)"]


def to_pil(t: torch.Tensor, upscale: int) -> Image.Image:
    arr = ((t.clamp(-1, 1) * 0.5 + 0.5) * 255).byte().permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    if upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
    return img


def make_sheet(rows: list[tuple[str, list[Image.Image]]]) -> Image.Image:
    tile = rows[0][1][0].width
    n_cols, n_rows = len(COL_NAMES), len(rows)
    sheet = Image.new(
        "RGB",
        (LABEL_W + n_cols * (tile + PAD), HEADER_H + n_rows * (tile + PAD)),
        (30, 30, 30),
    )
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(COL_NAMES):
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", help="directory of images")
    group.add_argument("--input-file", help="single image file")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--upscale", type=int, default=None,
                        help="nearest-neighbour tile upscale (default: 3 below 256px, else 1)")
    parser.add_argument("--name", default="contact_sheet")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    G = Generator(cfg["resolution"], cfg["base_channels"], cfg["max_channels"]).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    if args.upscale is None:
        args.upscale = 3 if cfg["resolution"] < 256 else 1

    from .data import build_transform
    tf = build_transform(cfg["resolution"], train=False)

    if args.input_file:
        paths = [Path(args.input_file)]
    else:
        ds = FlatImageDataset([args.input_dir], cfg["resolution"], train=False)
        paths = ds.paths[:args.limit] if args.limit else ds.paths

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in paths:
        x = tf(Image.open(p)).unsqueeze(0).to(device)
        out = G(x)[0]
        rows.append((p.stem, [to_pil(x[0], args.upscale), to_pil(out, args.upscale)]))

    sheet = make_sheet(rows)
    out_path = out_dir / f"{args.name}.png"
    sheet.save(out_path)
    print(f"saved {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
