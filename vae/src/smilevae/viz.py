"""Shared visualization helpers (no model dependency).

`to_pil` and `make_sheet` render tensors and labeled contact sheets. They are
used by every experiment (both the current SD-VAE method and the legacy
scratch VAE-GAN in ``old/``), so they live here rather than next to any one
model so importing them never pulls in a decoder.
"""

from PIL import Image, ImageDraw
import torch

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
