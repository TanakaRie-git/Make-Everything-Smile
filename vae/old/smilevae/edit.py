"""Apply the smile direction to object images at weak/mid/strong strengths.

For each input image: z = mu(E(x)); output D(z + alpha * proj_std * direction).
Saves a labeled contact sheet (rows = inputs; columns = input | recon |
weak | mid | strong), upscaled so low-resolution runs stay readable.

--mask restricts WHERE the direction is applied on the spatial latent grid
(spatial-latent checkpoints only), so only facial features — not the whole
face — possess the object:
    none        full direction map (default)
    eyes-mouth  geometric mask over the eye and mouth rows of the aligned
                CelebA layout
    top:<frac>  data-driven: keep the <frac> of latent cells with the largest
                direction magnitude (the smile signal concentrates at the
                eyes/mouth on its own), e.g. top:0.25

Usage:
    python -m smilevae.edit --ckpt outputs/hq512/ckpt/last.pt \
        --direction outputs/hq512/smile_direction.pt \
        --input-dir data/raw/objects_hq/golf_ball --out-dir outputs/hq512/edits \
        --strengths 4 8 12 --mask eyes-mouth
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .data import FlatImageDataset
from .models import VAEGAN

# fraction of the latent grid covered by eyes / mouth in aligned CelebA(-HQ)
EYE_ROWS = (0.32, 0.55)
MOUTH_ROWS = (0.58, 0.85)
FACE_COLS = (0.22, 0.78)


def gaussian_blur(mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Feather a (1,1,H,W) mask with a separable Gaussian."""
    radius = max(1, int(3 * sigma))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=mask.device)
    k = torch.exp(-(xs**2) / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    mask = F.conv2d(mask, k, padding=(0, radius))
    return F.conv2d(mask, k.transpose(2, 3), padding=(radius, 0))


def build_pixel_mask(size: int, device) -> torch.Tensor:
    """Feathered (1,1,size,size) mask over the eye and mouth bands — used to
    blend the smile delta into the object only where facial features belong,
    keeping the object's own pixels everywhere else."""
    mask = torch.zeros(1, 1, size, size, device=device)
    c0, c1 = (round(f * size) for f in FACE_COLS)
    for r0, r1 in (EYE_ROWS, MOUTH_ROWS):
        mask[:, :, round(r0 * size):round(r1 * size), c0:c1] = 1.0
    return gaussian_blur(mask, sigma=size / 24).clamp(0, 1)


def build_mask(direction: torch.Tensor, spec: str) -> torch.Tensor:
    """Return a (1, S, S) mask for a spatial direction map of shape (C, S, S)."""
    if spec == "none":
        return torch.ones(1, *direction.shape[1:], device=direction.device)
    if direction.dim() != 3:
        raise SystemExit(f"--mask {spec} requires a spatial latent, got shape {tuple(direction.shape)}")
    size = direction.shape[-1]
    if spec == "eyes-mouth":
        mask = torch.zeros(1, size, size, device=direction.device)
        c0, c1 = (round(f * size) for f in FACE_COLS)
        for r0, r1 in (EYE_ROWS, MOUTH_ROWS):
            mask[:, round(r0 * size):round(r1 * size), c0:c1] = 1.0
        return mask
    if spec.startswith("top:"):
        frac = float(spec.split(":", 1)[1])
        cell_norm = direction.norm(dim=0)
        k = max(1, round(frac * cell_norm.numel()))
        thresh = cell_norm.flatten().topk(k).values[-1]
        return (cell_norm >= thresh).float().unsqueeze(0)
    raise SystemExit(f"unknown --mask: {spec}")

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
    parser.add_argument("--upscale", type=int, default=None,
                        help="nearest-neighbor tile upscale (default: 3 below 256px, else 1)")
    parser.add_argument("--name", default="contact_sheet", help="output file stem")
    parser.add_argument("--mask", default="none", help="none | eyes-mouth | top:<frac>")
    parser.add_argument("--feature-only", action="store_true",
                        help="subtract the direction's spatial mean (the uniform "
                             "'become-a-face' shift), keeping only localized eye/mouth "
                             "structure — reduces contour/skin leaking onto the object")
    parser.add_argument("--blend", action="store_true",
                        help="pixel-space delta blend: output = input + pixel_mask * "
                             "(decode(z+dir) - decode(z)). Keeps the object's own sharp "
                             "pixels and overlays only the smile delta in the eye/mouth "
                             "region, at the object's own scale")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"],
                   cfg.get("latent_size", 0)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dir_state = torch.load(args.direction, map_location=device, weights_only=False)
    direction = dir_state["direction"].to(device)
    scale = dir_state["proj_std"]
    if args.feature_only:
        if direction.dim() != 3:
            raise SystemExit("--feature-only requires a spatial latent direction")
        direction = direction - direction.mean(dim=(1, 2), keepdim=True)
    direction = direction * build_mask(direction, args.mask)

    if args.upscale is None:
        args.upscale = 3 if cfg["resolution"] < 256 else 1

    ds = FlatImageDataset([args.input_dir], cfg["resolution"], train=False)
    n = min(len(ds), args.limit) if args.limit else len(ds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    col_names = ["input", "recon"] + [
        f"{nm} (α={a:g})" for nm, a in zip(STRENGTH_NAMES, args.strengths)
    ]
    pmask = build_pixel_mask(cfg["resolution"], device) if args.blend else None

    rows = []
    for i in range(n):
        x = ds[i].unsqueeze(0).to(device)
        mu, _ = model.encoder(x)
        rec = model.decoder(mu)
        outputs = [x[0], rec[0]]
        for alpha in args.strengths:
            edited = model.decoder(mu + alpha * scale * direction)
            if args.blend:
                edited = (x + pmask * (edited - rec)).clamp(-1, 1)
            outputs.append(edited[0])
        rows.append((ds.paths[i].stem, [to_pil(t, args.upscale) for t in outputs]))

    sheet = make_sheet(rows, col_names)
    out_path = out_dir / f"{args.name}.png"
    sheet.save(out_path)
    print(f"saved {n} rows -> {out_path} (columns: {' | '.join(col_names)})")


if __name__ == "__main__":
    main()
