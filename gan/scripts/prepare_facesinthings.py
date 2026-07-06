"""Prepare FacesInThings face-box crops for GAN pareidolia training.

Reads FacesInThings metadata.csv, crops the primary face rectangle from
each Neutral / Happy image, and saves them into:

    <out_dir>/neutral/   <- Emotion? == Neutral
    <out_dir>/happy/     <- Emotion? == Happy

These two directories are then referenced in configs/pareidolia256.yaml:
  - happy/  -> smile_dirs (is_face=True, label=1.0)
  - neutral/ -> neutral_dirs (is_face=True, label=0.0)

FacesInThings は「物体の中に見える顔」なので、画像全体を学習に使うと
顔領域が解像度縮小で潰れる。顔矩形でクロップしてから渡すことで
GANの識別器が感情を正しく学習できる。

クロップ規則は diffusion/data/build_pairs_pareidolia.py と同一:
  side = max(w, h) * (1 + box_pad) を正方形でクロップ、画像端でクランプ。
（学習クロップと推論クロップを一致させる。）

実行順序:
    uv run scripts/download_faces_in_things.py --out data/raw/faces_in_things
    uv run scripts/prepare_facesinthings.py \\
        --fit_dir data/raw/faces_in_things/FacesInThings \\
        --out_dir data/raw/faces_in_things/crops \\
        --split train
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

COL_FACE = "Is there a face?"
COL_EMOTION = "Emotion?"
COL_TRAIN = "train"
COL_IS_PRIMARY = "is_primary"


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _primary_box(boxes: list) -> list | None:
    """面積最大の矩形を代表顔として返す。"""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def crop_square_box(img: Image.Image, box: list, pad: float) -> Image.Image | None:
    """[x, y, w, h] の顔矩形を pad 割合ぶん広げて正方形でクロップする。

    diffusion/data/build_pairs_pareidolia.py の crop_square_box と同じ規則。
    """
    x1, y1, w, h = box
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    side = max(w, h) * (1.0 + pad)
    left = max(0, int(round(cx - side / 2.0)))
    top = max(0, int(round(cy - side / 2.0)))
    right = min(img.width, int(round(cx + side / 2.0)))
    bottom = min(img.height, int(round(cy + side / 2.0)))
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit_dir", required=True, type=Path,
                        help="download_faces_in_things.py が作った FacesInThings/ ディレクトリ"
                             "（images/ と metadata.csv を含む）")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="クロップ画像の出力先。<out_dir>/neutral/ と <out_dir>/happy/ を作る")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"],
                        help="使う split（test は評価用に温存するため既定は train）")
    parser.add_argument("--primary_only", action="store_true",
                        help="is_primary な行だけを使う")
    parser.add_argument("--box_pad", type=float, default=0.3,
                        help="顔矩形の余白割合（infer.py / diffusion track と揃える）")
    args = parser.parse_args()

    fit_dir = args.fit_dir.expanduser().resolve()
    meta_file = fit_dir / "metadata.csv"
    images_dir = fit_dir / "images"

    if not meta_file.is_file():
        raise FileNotFoundError(
            f"metadata.csv が見つかりません: {meta_file}\n"
            f"先に download_faces_in_things.py を実行してください。"
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images/ ディレクトリが見つかりません: {images_dir}")

    neutral_dir = args.out_dir / "neutral"
    happy_dir = args.out_dir / "happy"
    neutral_dir.mkdir(parents=True, exist_ok=True)
    happy_dir.mkdir(parents=True, exist_ok=True)

    with meta_file.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # フィルタリング
    targets = []
    for row in rows:
        if row.get(COL_FACE, "").strip().lower() not in {"yes", "several"}:
            continue
        is_train = _as_bool(row.get(COL_TRAIN))
        if args.split == "train" and not is_train:
            continue
        if args.split == "test" and is_train:
            continue
        if args.primary_only and not _as_bool(row.get(COL_IS_PRIMARY)):
            continue
        emo = row.get(COL_EMOTION, "").strip()
        if emo not in {"Neutral", "Happy"}:
            continue
        try:
            boxes = json.loads(row.get("boxes") or "[]")
        except (json.JSONDecodeError, TypeError):
            boxes = []
        box = _primary_box(boxes)
        if box is None:
            continue
        targets.append((row["file"], emo, box))

    n_neutral_target = sum(1 for _, e, _ in targets if e == "Neutral")
    n_happy_target = sum(1 for _, e, _ in targets if e == "Happy")
    print(f"[info] split={args.split}  Neutral={n_neutral_target}  Happy={n_happy_target}")

    counts = {"Neutral": 0, "Happy": 0}
    skipped = 0

    for file, emo, box in tqdm(targets, desc="cropping"):
        src = images_dir / file
        if not src.is_file():
            skipped += 1
            continue
        out_path = (neutral_dir if emo == "Neutral" else happy_dir) / f"{Path(file).stem}.png"
        if out_path.is_file():
            counts[emo] += 1
            continue
        try:
            img = Image.open(src).convert("RGB")
        except Exception:
            skipped += 1
            continue
        crop = crop_square_box(img, box, args.box_pad)
        if crop is None:
            skipped += 1
            continue
        crop.save(out_path)
        counts[emo] += 1

    print(f"[done] neutral={counts['Neutral']}  happy={counts['Happy']}"
          f"  skipped={skipped}")
    print(f"  -> {neutral_dir}")
    print(f"  -> {happy_dir}")


if __name__ == "__main__":
    main()
