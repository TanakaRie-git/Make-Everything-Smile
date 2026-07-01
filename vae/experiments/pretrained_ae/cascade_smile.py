"""2段カスケード: 「一度顔っぽくした画像」を人間の笑顔ポリシーに再入力する。

仮説(レビューコメント): パレイドリア物体を先に顔っぽくしておけば、もう顔に近いので
人間の笑顔方向(CelebA)がよく効き、笑顔を作りやすいのではないか。

これを検証するため、生成画像を**decode → 再encode**して次段に渡す真のカスケードにする
(潜在で足すだけの線形重ね合わせではない):
  z0   = encode(x)
  face = decode(z0 + α1 · proj1 · d1)          # Stage1: 顔っぽい画像(=「その画像」)
  zf   = encode(face)                           # その画像を"入力"として再エンコード
  out  = decode(zf + α2 · proj2 · d2)           # Stage2: 人間の笑顔ポリシーを α ラダーで

既定は d1=d2=CelebA 笑顔方向(smile_direction)。Stage1 で顔へ寄せ、Stage2 で笑顔を強める。
比較のため出力は input | face(Stage1) | Stage2 の α ラダー を横並びにする。

使い方(cd vae から):
    uv run python experiments/pretrained_ae/cascade_smile.py \
        --stage1-direction outputs/pretrained_ae/smile_direction.pt --stage1-alpha 2 \
        --stage2-direction outputs/pretrained_ae/smile_direction.pt --stage2-scales 0 1 2 3 \
        --input-dir data/diffusion_compare/inputs \
        --ids experiments/diffusion_compare/eval_ids.txt \
        --out-dir outputs/pretrained_ae/cascade
"""

import argparse
from pathlib import Path

import torch

from smilevae.data import FlatImageDataset
from smilevae.edit import to_pil, make_sheet
from sdvae import load_sdvae, encode, decode


def load_eval_ids(path):
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-direction", required=True)
    p.add_argument("--stage1-alpha", type=float, default=2.0,
                   help="Stage1(顔っぽくする)の強度。0 なら素の再構成を次段へ")
    p.add_argument("--stage2-direction", required=True)
    p.add_argument("--stage2-scales", type=float, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--input-dir", required=True)
    p.add_argument("--ids", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--max-images", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d1s = torch.load(args.stage1_direction, map_location=device, weights_only=False)
    d2s = torch.load(args.stage2_direction, map_location=device, weights_only=False)
    d1, p1 = d1s["direction"].to(device), d1s["proj_std"]
    d2, p2 = d2s["direction"].to(device), d2s["proj_std"]
    resolution = args.resolution or d1s.get("resolution", 512)
    model = args.model or d1s.get("model", "stabilityai/sd-vae-ft-mse")

    vae = load_sdvae(model, device)

    keep = load_eval_ids(args.ids)
    ds = FlatImageDataset([args.input_dir], resolution, train=False)
    idx_by_stem = {p_.stem: i for i, p_ in enumerate(ds.paths)}
    if keep is not None:
        order = [idx_by_stem[s] for s in keep if s in idx_by_stem]
    else:
        order = list(range(len(ds)))
    if args.max_images:
        order = order[:args.max_images]
    if not order:
        raise SystemExit("処理対象 0 件。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    upscale = 1 if resolution >= 256 else 3
    col_names = ["input", f"face (α1={args.stage1_alpha:g})"] + \
                [f"smile α={a:g}" for a in args.stage2_scales]

    overview_rows = []
    for i in order:
        x = ds[i].unsqueeze(0).to(device)
        stem = ds.paths[i].stem
        face = decode(vae, encode(vae, x) + args.stage1_alpha * p1 * d1)   # Stage1
        zf = encode(vae, face)                                             # 再encode
        tiles = [x[0], face[0]]
        for a2 in args.stage2_scales:                                      # Stage2
            tiles.append(decode(vae, zf + a2 * p2 * d2)[0])
        pil_tiles = [to_pil(t, upscale) for t in tiles]
        make_sheet([(stem, pil_tiles)], col_names).save(out_dir / f"{stem}_compare.png")
        pil_tiles[-1].save(out_dir / f"{stem}_smile.png")
        overview_rows.append((stem, pil_tiles))

    make_sheet(overview_rows, col_names).save(out_dir / "_overview.png")
    print(f"saved {len(order)} images -> {out_dir}")
    print(f"  columns: {' | '.join(col_names)}")


if __name__ == "__main__":
    main()
