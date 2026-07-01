"""Diffusion トラックと同一入力・同一プロトコルで VAE の笑顔付与を出力する。

Diffusion トラック(`diffusion/scripts/infer.py`)は FacesInThings のパレイドリア
crop を入力に、笑顔化した `<id>_smile.png` と入出力比較 `<id>_compare.png` を出す。
本スクリプトはその作法をミラーし、**同じ crop・同じ ID** に対して VAE-GAN の
潜在編集 `z = mu(E(x)); D(z + alpha * proj_std * direction)` を **scale ラダー**で
適用する(alpha=0 は recon=identity、Concept Slider の scale=0 に対応)。

各入力について:
  <out>/<id>_compare.png   input | alpha=0 | alpha=a1 | ...  (1枚に横並び)
  <out>/<id>_smile.png     最大 alpha の笑顔画像(diffusion の *_smile.png に対応)
  <out>/_overview.png      全 ID をまとめたコンタクトシート

2条件(計画 §0.2)は --ckpt / --direction を差し替えて2回回す:
  Baseline    : outputs/hq512/ckpt/last.pt        + outputs/hq512/smile_direction.pt
  +Pareidolia : outputs/pareidolia512/ckpt/last.pt + outputs/pareidolia512/fit_direction.pt

使い方(cd vae から):
  uv run python experiments/diffusion_compare/compare_facesinthings.py \
      --ckpt outputs/hq512/ckpt/last.pt \
      --direction outputs/hq512/smile_direction.pt \
      --input-dir data/diffusion_compare/inputs \
      --ids experiments/diffusion_compare/eval_ids.txt \
      --out-dir outputs/diffusion_compare/baseline \
      --scales 0 4 8 12
"""

import argparse
from pathlib import Path

import torch

from smilevae.data import FlatImageDataset
from smilevae.models import VAEGAN
# 描画・マスクは本編の edit.py と共有(重複実装しない)
from smilevae.edit import to_pil, make_sheet, build_mask, build_pixel_mask


def load_eval_ids(path: str | None) -> list[str] | None:
    """eval_ids.txt から ID 一覧を読む(各行 '<id> [emotion]'、# はコメント)。"""
    if path is None:
        return None
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line.split()[0])
    return ids


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--input-dir", required=True, help="crop 画像の入ったディレクトリ")
    parser.add_argument("--ids", default=None,
                        help="評価する ID を絞る eval_ids.txt(未指定なら input-dir 全件)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 4, 8, 12],
                        help="alpha ラダー(proj_std 単位)。先頭に 0 を置くと identity 確認になる")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--mask", default="none", help="none | eyes-mouth | top:<frac>")
    parser.add_argument("--feature-only", action="store_true",
                        help="方向の空間平均(一様な'顔になる'成分)を引き、局所的な目/口構造だけ残す")
    parser.add_argument("--blend", action="store_true",
                        help="画素空間差分ブレンド: input + pixel_mask*(decode(z+dir)-decode(z))")
    parser.add_argument("--upscale", type=int, default=None,
                        help="タイルの最近傍拡大(既定: 256px 未満は 3、以上は 1)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    model = VAEGAN(cfg["resolution"], cfg["z_dim"], cfg["base_channels"], cfg["max_channels"],
                   cfg.get("latent_size", 0)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    dir_state = torch.load(args.direction, map_location=device, weights_only=False)
    direction = dir_state["direction"].to(device)
    proj_std = dir_state["proj_std"]
    if args.feature_only:
        if direction.dim() != 3:
            raise SystemExit("--feature-only requires a spatial latent direction")
        direction = direction - direction.mean(dim=(1, 2), keepdim=True)
    direction = direction * build_mask(direction, args.mask)

    if args.upscale is None:
        args.upscale = 3 if cfg["resolution"] < 256 else 1

    # 入力を eval_ids で絞る(diffusion と同一 ID の順序を保つ)
    keep = load_eval_ids(args.ids)
    ds = FlatImageDataset([args.input_dir], cfg["resolution"], train=False)
    idx_by_stem = {p.stem: i for i, p in enumerate(ds.paths)}
    if keep is not None:
        order = [idx_by_stem[s] for s in keep if s in idx_by_stem]
        missing = [s for s in keep if s not in idx_by_stem]
        if missing:
            print(f"warn: input-dir に無い ID をスキップ: {missing}")
    else:
        order = list(range(len(ds)))
    if args.max_images:
        order = order[:args.max_images]
    if not order:
        raise SystemExit("処理対象の画像が 0 件。--input-dir / --ids を確認。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pmask = build_pixel_mask(cfg["resolution"], device) if args.blend else None
    col_names = ["input"] + [f"α={a:g}" for a in args.scales]

    overview_rows = []
    for i in order:
        x = ds[i].unsqueeze(0).to(device)
        stem = ds.paths[i].stem
        mu, _ = model.encoder(x)
        rec = model.decoder(mu)
        tiles = [x[0]]
        for alpha in args.scales:
            edited = model.decoder(mu + alpha * proj_std * direction)
            if args.blend:
                edited = (x + pmask * (edited - rec)).clamp(-1, 1)
            tiles.append(edited[0])
        pil_tiles = [to_pil(t, args.upscale) for t in tiles]

        # per-image: diffusion の <id>_compare.png / <id>_smile.png をミラー
        make_sheet([(stem, pil_tiles)], col_names).save(out_dir / f"{stem}_compare.png")
        pil_tiles[-1].save(out_dir / f"{stem}_smile.png")
        overview_rows.append((stem, pil_tiles))

    make_sheet(overview_rows, col_names).save(out_dir / "_overview.png")
    print(f"saved {len(order)} images -> {out_dir}")
    print(f"  per-image: <id>_compare.png (columns: {' | '.join(col_names)}), <id>_smile.png")
    print(f"  overview : _overview.png")


if __name__ == "__main__":
    main()
