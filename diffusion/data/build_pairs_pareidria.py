"""Build mixed (neutral -> happy) pairs for the *pareidolia* smile policy.

狙い:
  パレイドリア（物体に見える顔）を neutral → happy に変換する IP2P+LoRA policy を
  学習するためのペアデータを作る。学習ドメイン(物体の顔)と推論ドメインを揃えるため、
  2 つのソースを **混合** する:

    (1) CelebAMask-HQ : Smiling 属性で neutral(=非笑顔) と smile を分け、
        独立サンプリングで neutral -> smile ペアを作る（顔のリアルな笑顔の事前知識）。
    (2) FacesInThings : metadata.csv の `Emotion?` 列で Neutral / Happy を分け、
        独立サンプリングで neutral -> happy ペアを作る（パレイドリア・ドメイン本体）。
        画像は「物体写真の一部」なので、顔矩形(boxes)で正方形クロップして保存し、
        そのクロップをペアの original/target に使う（infer.py --use_box と整合させる）。

  CelebA と FacesInThings には「同一対象の neutral/happy ペア」が存在しないため、
  どちらも build_pairs.py と同じく **群ごとに独立サンプリングしてペア化** する。

出力（既存の dataset.py / train.py がそのまま読める形式）:
    <out_dir>/train.json
    <out_dir>/val.json
  各要素: {"original": <neutralパス>, "target": <happyパス>,
           "instruction": <編集指示>, "source": "celeba" | "facesinthings"}

設計メモ:
  - FacesInThings は train split のみ学習に使い、test split は推論評価用に温存する。
  - CelebA が量で FacesInThings を圧倒しないよう、既定で枚数上限を分けて持つ
    （--max_celeba_pairs / --max_fit_pairs）。混合比はここで調整する。
  - TODO(将来拡張): identity ベースのペアリング / FacesInThings の Sad,Angry など他感情を
    neutral 側に取り込む拡張。今は neutral->happy の動作確認を優先。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

# CelebA 側は既存ロジックを再利用する。
from data.build_pairs import (
    _resolve_attr_file,
    _resolve_img_dir,
    parse_attributes,
    split_by_smiling,
)

# neutral -> happy の編集指示。CelebA(人物) と FacesInThings(物体の顔) の両方で
# 自然に読めるよう "person" に限定しない言い回しを混ぜる。1 つの instruction が
# 1 つの「笑顔化」写像に対応するよう、語彙は smile/happy に揃える。
INSTRUCTIONS = [
    "make it smile",
    "make the face smile",
    "make them smile",
    "make it happy",
    "give it a happy smiling face",
]

# FacesInThings metadata の列名。
COL_FACE = "Is there a face?"
COL_EMOTION = "Emotion?"
COL_TRAIN = "train"


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def crop_square_box(img: Image.Image, box, pad: float):
    """[x, y, w, h] の顔矩形を pad 割合ぶん広げて正方形でクロップする。

    infer.py の crop_box と同じ規則（学習クロップと推論クロップを一致させる）。
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


def _primary_box(boxes):
    """面積最大の矩形を顔とみなす（boxes は [[x, y, w, h], ...]）。"""
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def load_facesinthings_groups(fit_dir: Path, split: str, primary_only: bool):
    """metadata.csv を読み、Neutral / Happy の行を返す。

    Returns: (neutral_rows, happy_rows) — 各要素は (file, box) のタプル。
    """
    meta_file = fit_dir / "metadata.csv"
    if not meta_file.is_file():
        raise FileNotFoundError(f"metadata.csv が見つかりません: {meta_file}")

    neutral, happy = [], []
    with meta_file.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            # 顔が写っている行のみ（Yes / Several）。
            if row.get(COL_FACE, "").strip().lower() not in {"yes", "several"}:
                continue
            is_train = _as_bool(row.get(COL_TRAIN))
            if split == "train" and not is_train:
                continue
            if split == "test" and is_train:
                continue
            if primary_only and not _as_bool(row.get("is_primary")):
                continue
            try:
                boxes = json.loads(row.get("boxes") or "[]")
            except (json.JSONDecodeError, TypeError):
                boxes = []
            box = _primary_box(boxes)
            if box is None:
                continue
            emo = row.get(COL_EMOTION, "").strip()
            if emo == "Neutral":
                neutral.append((row["file"], box))
            elif emo == "Happy":
                happy.append((row["file"], box))
    return neutral, happy


def cache_crops(rows, images_dir: Path, crop_dir: Path, pad: float):
    """各 (file, box) を正方形クロップして crop_dir に保存し、クロップパスのリストを返す。"""
    crop_dir.mkdir(parents=True, exist_ok=True)
    out = []
    missing = 0
    for file, box in rows:
        src = images_dir / file
        if not src.is_file():
            missing += 1
            continue
        crop_path = crop_dir / f"{Path(file).stem}.png"
        if not crop_path.is_file():
            try:
                img = Image.open(src).convert("RGB")
            except Exception:
                missing += 1
                continue
            crop = crop_square_box(img, box, pad)
            if crop is None:
                continue
            crop.save(crop_path)
        out.append(str(crop_path.resolve()))
    if missing:
        print(f"[warn] FacesInThings: 画像欠損/読込失敗 {missing} 件をスキップ")
    return out


def sample_pairs(neutral_paths, happy_paths, max_pairs, source, rng):
    """neutral 群と happy 群を独立サンプリングしてペアを作る。"""
    n = min(len(neutral_paths), len(happy_paths))
    if max_pairs is not None:
        n = min(n, max_pairs)
    if n == 0:
        return []
    src = rng.sample(neutral_paths, n)
    dst = rng.sample(happy_paths, n)
    return [
        {"original": o, "target": t, "instruction": rng.choice(INSTRUCTIONS), "source": source}
        for o, t in zip(src, dst)
    ]


def build_celeba_pairs(celeba_root: Path, max_pairs, rng):
    """CelebAMask-HQ の Smiling 属性から neutral -> smile ペアを作る。"""
    attr_file = _resolve_attr_file(celeba_root)
    img_dir = _resolve_img_dir(celeba_root)
    print(f"[info] CelebA 属性: {attr_file}")
    print(f"[info] CelebA 画像: {img_dir}")
    header, rows = parse_attributes(attr_file)
    smiling, non_smiling = split_by_smiling(header, rows, img_dir)
    print(f"[info] CelebA  smile={len(smiling)}  neutral={len(non_smiling)}")
    neutral_paths = [str((img_dir / f).resolve()) for f in non_smiling]
    happy_paths = [str((img_dir / f).resolve()) for f in smiling]
    return sample_pairs(neutral_paths, happy_paths, max_pairs, "celeba", rng)


def build_facesinthings_pairs(fit_root: Path, crop_dir: Path, split, primary_only,
                              pad, max_pairs, rng):
    """FacesInThings の Emotion? から neutral -> happy ペアを作る（顔クロップ）。"""
    fit_dir = fit_root / "FacesInThings"
    if not fit_dir.is_dir():
        raise FileNotFoundError(
            f"FacesInThings が見つかりません: {fit_dir}\n"
            f"--facesinthings_root には <root>/FacesInThings/ を含む親を指定してください。"
        )
    images_dir = fit_dir / "images"
    neutral_rows, happy_rows = load_facesinthings_groups(fit_dir, split, primary_only)
    print(f"[info] FacesInThings(split={split})  neutral={len(neutral_rows)}  happy={len(happy_rows)}")

    neutral_paths = cache_crops(neutral_rows, images_dir, crop_dir / "neutral", pad)
    happy_paths = cache_crops(happy_rows, images_dir, crop_dir / "happy", pad)
    print(f"[info] FacesInThings crops  neutral={len(neutral_paths)}  happy={len(happy_paths)} -> {crop_dir}")
    return sample_pairs(neutral_paths, happy_paths, max_pairs, "facesinthings", rng)


def main():
    p = argparse.ArgumentParser(description="Build mixed neutral->happy pairs for pareidolia smile policy")
    # --- CelebA ---
    p.add_argument("--celeba_root", type=Path, default=None,
                   help="CelebAMask-HQ のルート（省略すると CelebA を混ぜない）")
    p.add_argument("--max_celeba_pairs", type=int, default=2000,
                   help="CelebA から作るペア数の上限（混合比の調整用）")
    # --- FacesInThings ---
    p.add_argument("--facesinthings_root", type=Path, default=None,
                   help="<root>/FacesInThings/ を含む親（省略すると FacesInThings を混ぜない）")
    p.add_argument("--fit_split", type=str, default="train", choices=["all", "train", "test"],
                   help="FacesInThings のどの split をペア化に使うか（既定 train。test は評価用に温存）")
    p.add_argument("--primary_only", action="store_true",
                   help="is_primary な顔のみ使う")
    p.add_argument("--box_pad", type=float, default=0.3,
                   help="顔クロップの余白（矩形サイズに対する割合, infer.py と揃える）")
    p.add_argument("--max_fit_pairs", type=int, default=None,
                   help="FacesInThings から作るペア数の上限（既定 None=作れるだけ）")
    p.add_argument("--crop_dir", type=Path, default=Path("data/cache/facesinthings_crops"),
                   help="FacesInThings 顔クロップのキャッシュ先")
    # --- 共通 ---
    p.add_argument("--out_dir", type=Path, default=Path("pairs/pareidria"))
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.celeba_root and not args.facesinthings_root:
        p.error("--celeba_root か --facesinthings_root の少なくとも一方を指定してください。")

    rng = random.Random(args.seed)
    pairs = []

    if args.facesinthings_root:
        pairs += build_facesinthings_pairs(
            args.facesinthings_root.expanduser().resolve(),
            args.crop_dir.expanduser().resolve(),
            args.fit_split, args.primary_only, args.box_pad,
            args.max_fit_pairs, rng,
        )
    if args.celeba_root:
        pairs += build_celeba_pairs(
            args.celeba_root.expanduser().resolve(), args.max_celeba_pairs, rng,
        )

    if not pairs:
        raise RuntimeError("ペアが 0 件でした。データ配置と Emotion?/Smiling ラベルを確認してください。")

    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(json.dumps(train_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "val.json").write_text(json.dumps(val_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    n_fit = sum(1 for x in pairs if x["source"] == "facesinthings")
    n_celeba = sum(1 for x in pairs if x["source"] == "celeba")
    print(f"[done] total={len(pairs)} (facesinthings={n_fit}, celeba={n_celeba})  "
          f"train={len(train_pairs)} val={len(val_pairs)}")
    print(f"        {out_dir / 'train.json'}")
    print(f"        {out_dir / 'val.json'}")


if __name__ == "__main__":
    main()
