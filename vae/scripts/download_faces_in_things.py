"""Download the Faces in Things dataset (Hamilton et al., arXiv:2409.16143)
and extract the `happy` subset (plan §2.2: +Pareidolia training data and the
naturalness reference distribution).

Official zip: https://aka.ms/faces-dataset (images/ + metadata.csv).
The metadata column `Emotion?` holds the annotated emotion; rows whose primary
annotation is Happy are copied into <out>/happy/.

Usage:
    uv run scripts/download_faces_in_things.py --out data/raw/faces_in_things
"""

import argparse
import csv
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

URL = "https://aka.ms/faces-dataset"


def download(url: str, dest: Path) -> None:
    # write to .part first so an interrupted download is never mistaken for a
    # complete zip on rerun
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(part, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dataset_dir = out / "FacesInThings"

    if not dataset_dir.exists():
        zip_path = out / "FacesInThings.zip"
        if not zip_path.exists():
            download(URL, zip_path)
        print("extracting...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out)

    images_dir = dataset_dir / "images"
    happy_dir = out / "happy"
    happy_dir.mkdir(exist_ok=True)

    emotions: Counter[str] = Counter()
    copied = 0
    with open(dataset_dir / "metadata.csv", newline="") as f:
        for row in csv.DictReader(f):
            emotion = (row.get("Emotion?") or "").strip()
            emotions[emotion or "(none)"] += 1
            if emotion.lower() == "happy":
                src = images_dir / row["file"]
                dst = happy_dir / row["file"]
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1

    print(f"emotion counts: {dict(emotions.most_common())}")
    print(f"happy subset: {copied} new files -> {happy_dir}")


if __name__ == "__main__":
    main()
