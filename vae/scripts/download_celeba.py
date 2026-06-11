"""Download smiling / non-smiling face images into class folders.

Two sources (plan §0.3 says CelebA-HQ is the smile teacher; plain CelebA is a
lighter stand-in for low-resolution smoke tests):

  celeba    flwrlabs/celeba (aligned 178x218, 40 binary attributes, streamed)
  celeba-hq Ryan-sjtu/celebahq-caption (1024px CelebA-HQ + caption; an image is
            counted as smiling iff its caption mentions smiling)

Usage:
    uv run scripts/download_celeba.py --source celeba --n-per-class 2000 \
        --out ../data/raw/celeba
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def iter_labeled(source: str):
    """Yield (image, smiling, male). Gender lets the direction computation
    average male/female smile directions, mitigating gender entanglement."""
    if source == "celeba":
        ds = load_dataset("flwrlabs/celeba", "img_align+identity+attr",
                          split="train", streaming=True)
        for row in ds:
            yield row["image"], bool(row["Smiling"]), bool(row["Male"])
    elif source == "celeba-hq":
        ds = load_dataset("Ryan-sjtu/celebahq-caption", split="train", streaming=True)
        for row in ds:
            text = row["text"].lower()
            yield row["image"], "smil" in text, "she" not in text and "woman" not in text
    else:
        raise ValueError(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["celeba", "celeba-hq"], default="celeba")
    parser.add_argument("--n-per-class", type=int, default=2000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-side", type=int, default=None,
                        help="downscale so the longer side is at most this (saves disk)")
    args = parser.parse_args()

    out = Path(args.out)
    counts = {f"{s}_{g}": 0 for s in ("smile", "neutral") for g in ("male", "female")}
    for name in counts:
        (out / name).mkdir(parents=True, exist_ok=True)

    bar = tqdm(total=4 * args.n_per_class, desc=f"{args.source} -> {out}")
    for i, (img, smiling, male) in enumerate(iter_labeled(args.source)):
        cls = f"{'smile' if smiling else 'neutral'}_{'male' if male else 'female'}"
        if counts[cls] >= args.n_per_class:
            continue
        if args.max_side and max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side))
        img.convert("RGB").save(out / cls / f"{i:07d}.jpg", quality=95)
        counts[cls] += 1
        bar.update(1)
        if all(c >= args.n_per_class for c in counts.values()):
            break
    bar.close()
    print(f"done: {counts} -> {out}")


if __name__ == "__main__":
    main()
