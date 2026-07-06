"""Apply the smile to object images and produce a labelled contact sheet.

Two editing modes (--mode):

  conditional (default)
    G(x, s) for s in strengths.  The generator's AdaIN conditioning directly
    controls smile intensity.  This is the GAN track's primary output.

  latent
    Encode → perturb spatial latent by alpha * proj_std * direction → decode
    (via G.decode_raw, bypassing the AdaIN blocks).  Mirrors the VAE track's
    edit path so the two tracks' latent representations can be compared.
    Requires a pre-computed direction from direction.py (--direction).

For both modes the contact sheet layout mirrors vae/edit.py:
  rows = input images, columns = input | recon | weak | mid | strong

Usage — conditional (primary):
    python -m smilegan.edit --ckpt outputs/smoke64_gan/ckpt/last.pt \
        --input-dir data/raw/objects_smoke --out-dir outputs/smoke64_gan/edits \
        --strengths 0.5 1.0 2.0

Usage — latent (cross-track comparison):
    python -m smilegan.edit --ckpt outputs/smoke64_gan/ckpt/last.pt \
        --direction outputs/smoke64_gan/smile_direction.pt \
        --mode latent --strengths 1.0 2.0 3.0 \
        --input-dir data/raw/objects_smoke --out-dir outputs/smoke64_gan/edits_latent
"""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .data import FlatImageDataset
from .models import Generator

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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", help="directory of images")
    group.add_argument("--input-file", help="single image file")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["conditional", "latent"], default="conditional")
    parser.add_argument("--direction",
                        help="path to direction .pt from direction.py (required for --mode latent)")
    parser.add_argument("--strengths", type=float, nargs=3, default=[0.5, 1.0, 2.0],
                        help="weak/mid/strong smile values. "
                             "Conditional: direct smile scalar (0=neutral, 1=trained max). "
                             "Latent: in units of proj_std (matches VAE convention).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--upscale", type=int, default=None,
                        help="nearest-neighbour tile upscale (default: 3 below 256px, else 1)")
    parser.add_argument("--name", default="contact_sheet")
    parser.add_argument("--single-strength", type=float, default=None,
                        help="if set, save only G(x, s) at this strength as a plain image")
    args = parser.parse_args()

    if args.mode == "latent" and args.direction is None:
        raise SystemExit("--direction is required for --mode latent")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    G = Generator(
        cfg["resolution"], cfg["z_channels"], cfg["latent_size"],
        cfg["base_channels"], cfg["max_channels"], cfg["n_res"], cfg["style_dim"],
    ).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    direction, scale = None, 1.0
    if args.mode == "latent":
        dir_state = torch.load(args.direction, map_location=device, weights_only=False)
        direction = dir_state["direction"].to(device)
        scale = dir_state["proj_std"]

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

    col_names = ["input", "recon"] + [
        f"{nm} ({'s' if args.mode == 'conditional' else 'α'}={a:g})"
        for nm, a in zip(STRENGTH_NAMES, args.strengths)
    ]
    rows = []
    for p in paths:
        x = tf(Image.open(p)).unsqueeze(0).to(device)
        if args.single_strength is not None:
            s = x.new_full((1,), args.single_strength)
            out = G(x, s)[0]
            out_path = out_dir / f"{args.name}.png"
            to_pil(out, args.upscale).save(out_path)
            print(f"saved -> {out_path} (s={args.single_strength})")
            return
        if args.mode == "conditional":
            # recon = G(x, 0.0) to verify reconstruction quality
            recon = G(x, x.new_zeros(1))
            outputs = [x[0], recon[0]]
            for s in args.strengths:
                smile = x.new_full((1,), s)
                outputs.append(G(x, smile)[0])
        else:
            # latent mode: encode → perturb → decode_raw (bypasses AdaIN)
            z = G.encode(x)
            recon = G.decode_raw(z)
            outputs = [x[0], recon[0]]
            for alpha in args.strengths:
                z_edited = z + alpha * scale * direction.unsqueeze(0)
                outputs.append(G.decode_raw(z_edited)[0])
        rows.append((p.stem, [to_pil(t, args.upscale) for t in outputs]))

    sheet = make_sheet(rows, col_names)
    out_path = out_dir / f"{args.name}.png"
    sheet.save(out_path)
    print(f"saved {len(rows)} rows -> {out_path} (columns: {' | '.join(col_names)})")

    smiles_dir = out_dir / "smiles"
    smiles_dir.mkdir(exist_ok=True)
    for stem, tiles in rows:
        tiles[-1].save(smiles_dir / f"{stem}_smile.png")
    print(f"saved individual smile images -> {smiles_dir}/")


if __name__ == "__main__":
    main()
