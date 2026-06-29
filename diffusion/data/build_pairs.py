"""Build (non-smiling -> smiling) image pairs from CelebA-HQ for InstructPix2Pix training.

設計判断（引き継ぎ書より）:
  CelebA-HQ には同一人物の笑顔/非笑顔ペアが存在しない。よって `Smiling` 属性で
  画像を二群に分け、非笑顔(smiling=0)と笑顔(smiling=1)を **別々にランダムサンプリング**
  してペアを作る。生成物は学習に使う JSON ファイル:

      <out_dir>/train.json
      <out_dir>/val.json

  各 JSON は次の形式のリスト:
      [{"original": "<非笑顔の絶対パス>",
        "target":   "<笑顔の絶対パス>",
        "instruction": "make the person smile"}, ...]

  `original` を入力（非笑顔）、`target` を教師（笑顔）として IP2P を学習する。

TODO(将来拡張): 品質が低い場合は identity_CelebA.txt /
  CelebA-HQ-to-CelebA-mapping.txt を使い、同一人物内でのペアリングへ変更する。
  今は属性ベースのランダムペアリングで動作確認を優先する。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# 学習時の編集指示文（IP2P の instruction）。バリエーションを足すと頑健になる。
INSTRUCTIONS = [
    "make the person smile",
    "make this person smile",
    "give the person a smile",
    "make them smile",
]


def _resolve_attr_file(data_root: Path) -> Path:
    """属性ファイルのパスを探す。複数の慣例的な配置に対応する。"""
    candidates = [
        data_root / "CelebAMask-HQ-attribute_list.txt",
        data_root / "CelebAMask-HQ-attribute-anno.txt",
        data_root / "list_attr_celeba.txt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"属性ファイルが見つかりません。次のいずれかを {data_root} に配置してください: "
        + ", ".join(c.name for c in candidates)
    )


def _resolve_img_dir(data_root: Path) -> Path:
    """画像ディレクトリを探す。"""
    candidates = [
        data_root / "CelebA-HQ-img",
        data_root / "CelebAMask-HQ-img",
        data_root / "images",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"画像ディレクトリが見つかりません。次のいずれかを {data_root} に配置してください: "
        + ", ".join(c.name for c in candidates)
    )


def parse_attributes(attr_file: Path):
    """CelebA(-HQ) 属性ファイルをパースする。

    形式:
        行1: 画像枚数 (例: 30000)  ※ない場合もあるので柔軟に扱う
        行2: 属性名（スペース区切り）
        行3-: <filename> <v1> <v2> ... （値は 1 / -1）

    Returns:
        (header: list[str], rows: dict[filename -> list[int]])
    """
    with attr_file.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # 1行目が数値だけならヘッダはその次の行。
    first_tokens = lines[0].split()
    if len(first_tokens) == 1 and first_tokens[0].isdigit():
        header = lines[1].split()
        data_lines = lines[2:]
    else:
        header = first_tokens
        data_lines = lines[1:]

    rows = {}
    for ln in data_lines:
        parts = ln.split()
        fname = parts[0]
        values = [int(v) for v in parts[1:]]
        rows[fname] = values
    return header, rows


def split_by_smiling(header, rows, img_dir: Path):
    """Smiling 属性で笑顔 / 非笑顔のファイル名リストへ分割する。"""
    if "Smiling" not in header:
        raise ValueError(f"属性ファイルに 'Smiling' 列がありません。列: {header}")
    smile_idx = header.index("Smiling")

    smiling, non_smiling = [], []
    missing = 0
    for fname, values in rows.items():
        if not (img_dir / fname).is_file():
            missing += 1
            continue
        if values[smile_idx] == 1:
            smiling.append(fname)
        else:
            non_smiling.append(fname)
    if missing:
        print(f"[warn] 属性に存在するが画像が見つからないファイル: {missing} 件をスキップ")
    return smiling, non_smiling


def make_pairs(non_smiling, smiling, img_dir: Path, max_pairs: int, rng: random.Random):
    """非笑顔と笑顔を独立サンプリングしてペアを作る。"""
    n = min(max_pairs, len(non_smiling), len(smiling))
    if n == 0:
        raise RuntimeError(
            f"ペアを作れません（非笑顔={len(non_smiling)}, 笑顔={len(smiling)}）。"
        )
    src = rng.sample(non_smiling, n)
    dst = rng.sample(smiling, n)
    pairs = []
    for orig, tgt in zip(src, dst):
        pairs.append(
            {
                "original": str((img_dir / orig).resolve()),
                "target": str((img_dir / tgt).resolve()),
                "instruction": rng.choice(INSTRUCTIONS),
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Build smile pairs for IP2P training")
    parser.add_argument("--data_root", required=True, type=Path,
                        help="CelebA-HQ のルート（画像と属性ファイルを含む）")
    parser.add_argument("--out_dir", type=Path, default=Path("pairs"),
                        help="train.json / val.json の出力先（デフォルト: ./pairs）")
    parser.add_argument("--max_pairs", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    data_root = args.data_root.expanduser().resolve()
    attr_file = _resolve_attr_file(data_root)
    img_dir = _resolve_img_dir(data_root)
    print(f"[info] 属性ファイル: {attr_file}")
    print(f"[info] 画像ディレクトリ: {img_dir}")

    header, rows = parse_attributes(attr_file)
    smiling, non_smiling = split_by_smiling(header, rows, img_dir)
    print(f"[info] 笑顔: {len(smiling)} 枚 / 非笑顔: {len(non_smiling)} 枚")

    pairs = make_pairs(non_smiling, smiling, img_dir, args.max_pairs, rng)
    rng.shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text(
        json.dumps(train_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "val.json").write_text(
        json.dumps(val_pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] train={len(train_pairs)} val={len(val_pairs)} -> {out_dir}")
    print(f"        {out_dir / 'train.json'}")
    print(f"        {out_dir / 'val.json'}")


if __name__ == "__main__":
    main()
