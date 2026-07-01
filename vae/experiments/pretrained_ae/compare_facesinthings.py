"""事前学習 SD-VAE を土台に、diffusion_compare と同一入力・同一プロトコルで
笑顔付与を出力する(スクラッチ VAE-GAN 版 experiments/diffusion_compare/ と対照)。

編集は z = encode(x); decode(z + alpha * proj_std * direction) を scale ラダーで。
decoder は事前学習・凍結の SD-VAE なので、スクラッチ版より再構成がシャープなはず。

使い方(cd vae から):
    uv run python experiments/pretrained_ae/compare_facesinthings.py \
        --direction outputs/pretrained_ae/smile_direction.pt \
        --input-dir data/diffusion_compare/inputs \
        --ids experiments/diffusion_compare/eval_ids.txt \
        --out-dir outputs/pretrained_ae/compare --scales 0 1 2 3
"""

import argparse
from pathlib import Path

import torch

from smilevae.data import FlatImageDataset
from smilevae.edit import to_pil, make_sheet          # 描画は共有ヘルパを再利用
from sdvae import load_sdvae, encode, decode


def keep_input_color(edited: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """編集後の輝度(Y)は残し、色差(Cb/Cr)は入力から取る。全体の色ドリフトを
    decode 後に打ち消す(笑顔の陰影・歯の明るさは輝度側なので保たれる)。入出力は
    (C,H,W) の [-1,1]。"""
    e = edited * 0.5 + 0.5
    i = inp * 0.5 + 0.5
    ey = 0.299 * e[0] + 0.587 * e[1] + 0.114 * e[2]
    icb = -0.168736 * i[0] - 0.331264 * i[1] + 0.5 * i[2] + 0.5
    icr = 0.5 * i[0] - 0.418688 * i[1] - 0.081312 * i[2] + 0.5
    r = ey + 1.402 * (icr - 0.5)
    g = ey - 0.344136 * (icb - 0.5) - 0.714136 * (icr - 0.5)
    b = ey + 1.772 * (icb - 0.5)
    return (torch.stack([r, g, b]).clamp(0, 1) * 2 - 1)


def load_eval_ids(path: str | None) -> list[str] | None:
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
    parser.add_argument("--direction", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 1, 2, 3],
                        help="alpha ラダー(proj_std 単位)。先頭 0 は recon=identity")
    parser.add_argument("--resolution", type=int, default=None,
                        help="既定は direction 保存時の解像度")
    parser.add_argument("--model", default=None, help="既定は direction 保存時のモデル")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--feature-only", action="store_true",
                        help="方向の空間平均(=全体の色/トーンを一律に押す成分)を引き、"
                             "局所的な表情構造だけ残す。色ドリフト抑制。")
    parser.add_argument("--keep-color", action="store_true",
                        help="出力の色を入力に合わせ直す(輝度は編集後、色相・彩度は入力)。"
                             "色ドリフトを decode 後に打ち消す。")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dir_state = torch.load(args.direction, map_location=device, weights_only=False)
    direction = dir_state["direction"].to(device)
    proj_std = dir_state["proj_std"]
    if args.feature_only:
        # 空間平均(チャネルごとの一様成分)= 全体の色・トーン押し。これを除去。
        direction = direction - direction.mean(dim=(1, 2), keepdim=True)
    resolution = args.resolution or dir_state.get("resolution", 512)
    model = args.model or dir_state.get("model", "stabilityai/sd-vae-ft-mse")

    vae = load_sdvae(model, device)

    keep = load_eval_ids(args.ids)
    ds = FlatImageDataset([args.input_dir], resolution, train=False)
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
        raise SystemExit("処理対象 0 件。--input-dir / --ids を確認。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    col_names = ["input"] + [f"α={a:g}" for a in args.scales]
    upscale = 1 if resolution >= 256 else 3

    overview_rows = []
    for i in order:
        x = ds[i].unsqueeze(0).to(device)
        stem = ds.paths[i].stem
        z = encode(vae, x)
        tiles = [x[0]]
        for alpha in args.scales:
            edited = decode(vae, z + alpha * proj_std * direction)[0]
            if args.keep_color:
                edited = keep_input_color(edited, x[0])
            tiles.append(edited)
        pil_tiles = [to_pil(t, upscale) for t in tiles]
        make_sheet([(stem, pil_tiles)], col_names).save(out_dir / f"{stem}_compare.png")
        pil_tiles[-1].save(out_dir / f"{stem}_smile.png")
        overview_rows.append((stem, pil_tiles))

    make_sheet(overview_rows, col_names).save(out_dir / "_overview.png")
    print(f"saved {len(order)} images -> {out_dir}")
    print(f"  columns: {' | '.join(col_names)}")


if __name__ == "__main__":
    main()
