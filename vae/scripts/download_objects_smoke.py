"""Object images for the smoke test: Tiny ImageNet (64x64), filtered to
non-living classes via the WordNet hierarchy — the same animal/person/plant
exclusion the plan (§2.1) prescribes for the full ImageNet curation, so this
script doubles as a dry run of that logic.

NOTE: this is a smoke-test stand-in. The full-resolution object set comes from
ImageNet proper, curated by the cross-cutting data owner.

Usage:
    uv run scripts/download_objects_smoke.py --out ../data/raw/objects_smoke \
        --n-classes 30 --n-per-class 100
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import nltk
from tqdm import tqdm

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
EXCLUDED_ROOTS = ["animal.n.01", "person.n.01", "plant.n.02", "plant_part.n.01", "fungus.n.01"]


def is_nonliving(wnid: str) -> bool:
    from nltk.corpus import wordnet as wn

    synset = wn.synset_from_pos_and_offset("n", int(wnid[1:]))
    hypernyms = {h.name() for path in synset.hypernym_paths() for h in path}
    return not any(root in hypernyms for root in EXCLUDED_ROOTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-classes", type=int, default=30)
    parser.add_argument("--n-per-class", type=int, default=100)
    args = parser.parse_args()

    nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet as wn

    out = Path(args.out)
    cache = out / "_tiny-imagenet-200"
    out.mkdir(parents=True, exist_ok=True)

    if not cache.exists():
        zip_path = out / "tiny-imagenet-200.zip"
        if not zip_path.exists():
            print(f"downloading {URL} (~240MB)...")
            urllib.request.urlretrieve(URL, zip_path)
        print("extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)
        (out / "tiny-imagenet-200").rename(cache)

    wnids = (cache / "wnids.txt").read_text().split()
    nonliving = [w for w in wnids if is_nonliving(w)]
    print(f"{len(nonliving)}/{len(wnids)} Tiny ImageNet classes are non-living")

    chosen = nonliving[: args.n_classes]
    total = 0
    for wnid in tqdm(chosen, desc="copying"):
        name = wn.synset_from_pos_and_offset("n", int(wnid[1:])).lemmas()[0].name()
        dst_dir = out / f"{wnid}_{name}"
        dst_dir.mkdir(exist_ok=True)
        for src in sorted((cache / "train" / wnid / "images").glob("*.JPEG"))[: args.n_per_class]:
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            total += 1
    print(f"done: {len(chosen)} classes, {total} images -> {out}")


if __name__ == "__main__":
    main()
