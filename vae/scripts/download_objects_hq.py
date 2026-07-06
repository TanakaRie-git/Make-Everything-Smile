"""High-resolution object images for the 512px runs: Imagenette (fastai's
full-size ImageNet subset, Apache-2.0), keeping only its 8 non-living classes.

Like Tiny ImageNet for the 64px smoke, this is a stand-in until the team's
curated ImageNet object set is ready (plan §2.1).

Usage:
    uv run scripts/download_objects_hq.py --out data/raw/objects_hq --max-side 600
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image
from tqdm import tqdm

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
# wnid -> name; tench (n01440764) and English springer (n02102040) excluded as living
CLASSES = {
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-per-class", type=int, default=600)
    parser.add_argument("--max-side", type=int, default=600,
                        help="downscale so the longer side is at most this (saves disk)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out.parent / "_cache"  # outside out/: training globs out/ recursively
    cache = cache_root / "imagenette2"

    if not cache.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
        tgz = cache_root / "imagenette2.tgz"
        if not tgz.exists():
            print(f"downloading {URL} (~1.5GB)...")
            part = tgz.with_suffix(".tgz.part")
            urllib.request.urlretrieve(URL, part)
            part.rename(tgz)
        print("extracting...")
        with tarfile.open(tgz) as tf:
            tf.extractall(cache_root)

    for wnid, name in tqdm(CLASSES.items(), desc="resizing"):
        cls_dir = out / name
        cls_dir.mkdir(exist_ok=True)
        srcs = sorted((cache / "train" / wnid).glob("*.JPEG"))[: args.n_per_class]
        for src in srcs:
            dst = cls_dir / (src.stem + ".jpg")
            if dst.exists():
                continue
            img = Image.open(src).convert("RGB")
            if max(img.size) > args.max_side:
                img.thumbnail((args.max_side, args.max_side))
            img.save(dst, quality=92)
    counts = {name: len(list((out / name).iterdir())) for name in CLASSES.values()}
    print(f"done: {counts} -> {out}")


if __name__ == "__main__":
    main()
