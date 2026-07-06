"""3手法共通の軽量評価(10枚でも回る叩き台)。

各 (入力 crop, 出力笑顔画像) ペアに対し:
  Preservation : LPIPS↓ / SSIM↑ / CLIP画像類似↑(入力↔出力の意味的近さ=物体保存)
  Smile        : CLIP-smile(出力) と その入力からの増分 Δ↑
                 (= cos(img,"smiling") - cos(img,"neutral"))
を計算し、変種ごとに平均して表示する。Naturalness(FID/KID)は数百枚必要なので別途。

CLIP-smile は実顔分類器と違いパレイドリア物体にも反応しやすい(AU 検出より頑健)。
CLIP画像類似は「人間化すると下がる」ので物体アイデンティティ保持の代理になる。

使い方(cd gan から):
    uv run python experiments/eval.py \
      --variant scratch_baseline data/diffusion_compare/inputs outputs/diffusion_compare/baseline \
      --variant sdvae_fit_kc      data/diffusion_compare/inputs outputs/pretrained_ae/compare_fit_kc \
      --out outputs/eval/metrics.csv
各 --variant は「LABEL 入力dir 出力dir」。出力dir 内の <id>_smile.png を入力 <id>.* と対応させる。
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim
import lpips
import open_clip

SMILE_PROMPTS = ["a photo of a smiling face", "a happy face with a big smile"]
NEUTRAL_PROMPTS = ["a photo of a neutral face", "a face with a neutral expression"]


def load_rgb(path: Path, size: int) -> torch.Tensor:
    """[-1,1], (1,3,size,size)。白パディングで正方形化してから resize。"""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    t = transforms.functional.to_tensor(canvas.resize((size, size), Image.BICUBIC))
    return (t * 2 - 1).unsqueeze(0)


@torch.no_grad()
def clip_smile_scores(clip_model, imgs_pp, text_smile, text_neutral):
    """各画像の smile スコア = cos(img, mean_smile) - cos(img, mean_neutral)。"""
    feats = clip_model.encode_image(imgs_pp)
    feats = F.normalize(feats, dim=-1)
    return (feats @ text_smile) - (feats @ text_neutral)  # (N,)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", nargs=3, action="append", required=True,
                    metavar=("LABEL", "INPUT_DIR", "OUTPUT_DIR"))
    ap.add_argument("--size", type=int, default=256, help="LPIPS/SSIM の比較解像度")
    ap.add_argument("--out", default=None, help="CSV 保存先")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    clip_model, _, clip_pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_model = clip_model.to(device).eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        ts = F.normalize(clip_model.encode_text(tok(SMILE_PROMPTS).to(device)), dim=-1).mean(0)
        tn = F.normalize(clip_model.encode_text(tok(NEUTRAL_PROMPTS).to(device)), dim=-1).mean(0)
        ts, tn = F.normalize(ts, dim=-1), F.normalize(tn, dim=-1)

    rows = []
    header = ["variant", "n", "LPIPS↓", "SSIM↑", "CLIPimg↑", "CLIPsmile_out↑", "CLIPsmile_Δ↑"]
    for label, in_dir, out_dir in args.variant:
        in_dir, out_dir = Path(in_dir), Path(out_dir)
        L, S, CI, CSo, CSd, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for smile_path in sorted(out_dir.glob("*_smile.png")):
            stem = smile_path.stem.replace("_smile", "")
            src = next(in_dir.glob(f"{stem}.*"), None)
            if src is None:
                continue
            xi = load_rgb(src, args.size).to(device)
            xo = load_rgb(smile_path, args.size).to(device)
            L += lpips_fn(xi, xo).item()
            S += ssim((xo * 0.5 + 0.5), (xi * 0.5 + 0.5), data_range=1.0).item()
            # CLIP
            pil_i = Image.open(src).convert("RGB")
            pil_o = Image.open(smile_path).convert("RGB")
            batch = torch.stack([clip_pp(pil_i), clip_pp(pil_o)]).to(device)
            fe = F.normalize(clip_model.encode_image(batch), dim=-1)
            CI += (fe[0] @ fe[1]).item()
            smile_i = (fe[0] @ ts) - (fe[0] @ tn)
            smile_o = (fe[1] @ ts) - (fe[1] @ tn)
            CSo += smile_o.item()
            CSd += (smile_o - smile_i).item()
            n += 1
        if n:
            rows.append([label, n, L / n, S / n, CI / n, CSo / n, CSd / n])

    # 表示(まず全セルを文字列化してから桁揃え)
    def cell(v, i):
        return str(v) if i < 2 else f"{v:.4f}"
    str_rows = [[cell(v, i) for i, v in enumerate(r)] for r in rows]
    widths = [max(len(header[i]), *(len(sr[i]) for sr in str_rows)) if str_rows else len(header[i])
              for i in range(len(header))]
    print(" | ".join(header[i].rjust(widths[i]) for i in range(len(header))))
    print("-+-".join("-" * w for w in widths))
    for sr in str_rows:
        print(" | ".join(sr[i].rjust(widths[i]) for i in range(len(header))))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(v) if i < 2 else f"{v:.5f}" for i, v in enumerate(r)) + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()