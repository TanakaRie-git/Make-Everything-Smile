"""Single, large-in-frame objects (e.g. fruit) for pareidolia editing demos.

Buildings/scenes (Imagenette church, gas_pump) put the object inside a wider
scene, so a centered face overlay lands on background. Single centered objects
that fill the frame — like the Faces in Things apple/bag examples — are the
right canvas. Source: VinayHajare/Fruits-30 (single fruit per image, 400-1300px).

Usage:
    uv run scripts/download_objects_single.py --out data/raw/objects_single \
        --per-class 6 --max-side 600
"""

import argparse
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--max-side", type=int, default=600)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("VinayHajare/Fruits-30", split="train", streaming=True)
    names = ds.features["label"].names

    counts: Counter[int] = Counter()
    for i, row in enumerate(tqdm(ds, desc=f"fruits-30 -> {out}")):
        label = row["label"]
        if counts[label] >= args.per_class:
            continue
        img = row["image"].convert("RGB")
        if min(img.size) < 200:  # skip thumbnails; we want large single objects
            continue
        if max(img.size) > args.max_side:
            img.thumbnail((args.max_side, args.max_side))
        cls_dir = out / names[label].replace(" ", "_")
        cls_dir.mkdir(exist_ok=True)
        img.save(cls_dir / f"{i:06d}.jpg", quality=92)
        counts[label] += 1

    total = sum(counts.values())
    print(f"done: {total} images across {len(counts)} classes -> {out}")


if __name__ == "__main__":
    main()
