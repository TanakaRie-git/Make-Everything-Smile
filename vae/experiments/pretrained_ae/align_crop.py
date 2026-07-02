"""FacesInThings の顔 box を CelebA 準拠の整列フレームに正規化してクロップする。

「笑顔が毎回同じ位置に載る」対策(1)。笑顔方向は整列済み CelebA から作った固定空間
マップなので、入力側も **顔が frame 内の同じ位置・同じ占有率**に来るよう揃える。
ランドマークは無い(FacesInThings は box のみ)ので、box の中心・スケールで正規化する:

  side = max(bw, bh) / FACE_FRAC        # 顔 box が frame の FACE_FRAC を占めるスケール
  crop = 原画像から、box 中心が出力の (0.5, V_CENTER) に来る side×side を切り出す

使い方(cd vae から):
    uv run python experiments/pretrained_ae/align_crop.py \
        --meta data/raw/faces_in_things/FacesInThings/metadata.csv \
        --images data/raw/faces_in_things/FacesInThings/images \
        --ids experiments/eval_ids.txt \
        --out-dir data/diffusion_compare/inputs_aligned
"""

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def primary_box(boxes):
    """面積最大の box [x, y, w, h] を主顔とみなす。"""
    return max(boxes, key=lambda b: b[2] * b[3]) if boxes else None


def load_ids(path):
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


def align_one(img: Image.Image, box, face_frac: float, v_center: float) -> Image.Image:
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) / face_frac
    left = cx - 0.5 * side
    top = cy - v_center * side
    # 白キャンバスに重なり領域を貼る(はみ出しは白パディング=data.py と作法統一)。
    canvas = Image.new("RGB", (round(side), round(side)), (255, 255, 255))
    src = img.convert("RGB")
    ox, oy = round(-left), round(-top)          # canvas 上での原画像左上位置
    canvas.paste(src, (ox, oy))
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--face-frac", type=float, default=0.62,
                   help="顔 box が出力フレームで占める割合(小さいほど余白大)")
    p.add_argument("--v-center", type=float, default=0.46,
                   help="box 中心の縦位置(小さいほど顔が上=口が下の canonical 行へ)")
    args = p.parse_args()

    want = set(load_ids(args.ids))
    boxes_by_id = {}
    with open(args.meta) as f:
        for row in csv.DictReader(f):
            stem = Path(row["file"]).stem
            if stem in want:
                try:
                    boxes_by_id[stem] = json.loads(row.get("boxes") or "[]")
                except json.JSONDecodeError:
                    boxes_by_id[stem] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = Path(args.images)
    n = 0
    for stem in load_ids(args.ids):
        box = primary_box(boxes_by_id.get(stem, []))
        if box is None:
            print(f"skip {stem}: box なし")
            continue
        src = next(images.glob(f"{stem}.*"), None)
        if src is None:
            print(f"skip {stem}: 画像なし")
            continue
        aligned = align_one(Image.open(src), box, args.face_frac, args.v_center)
        aligned.save(out_dir / f"{stem}.png")
        n += 1
    print(f"aligned {n} images -> {out_dir} (face_frac={args.face_frac}, v_center={args.v_center})")


if __name__ == "__main__":
    main()
