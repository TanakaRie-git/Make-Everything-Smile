"""step015000.png と同じ3行グリッドを任意の人物画像で生成する。

グリッド構成:
  行1: 入力画像
  行2: G(x, source_label)  再構成
  行3: G(x, target_label)  変換（neutral→smile または smile→neutral）

Usage:
  uv run python scripts/make_human_sample.py \
      --ckpt outputs/pareidolia256_gan/ckpt/last.pt \
      --images path/a.jpg path/b.jpg path/c.jpg \
      --labels 0.0 1.0 0.0 \
      --out outputs/pareidolia256_gan/samples/human3.png

  --labels: 各画像の smile ラベル (0.0=neutral, 1.0=smile)
  target は自動的に 1-label になる（neutral→smile, smile→neutral）
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from smilegan.models import Generator
from torchvision import transforms as T


def build_crop_transform(resolution: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB")),
        T.Lambda(lambda im: T.CenterCrop(min(im.size))(im)),
        T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--images", nargs="+", required=True, help="画像パス（3枚推奨）")
    parser.add_argument("--labels", nargs="+", type=float, default=None,
                        help="各画像の smile ラベル (0.0=neutral, 1.0=smile)。省略時は全て 0.0")
    parser.add_argument("--out", default=None, help="出力 PNG パス。省略時は自動生成")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [0.0] * len(args.images)
    if args.out is None:
        stem = Path(args.images[0]).stem
        args.out = f"outputs/samples/{stem}_fake.png"
    if len(args.images) != len(args.labels):
        raise SystemExit("--images と --labels の数を合わせてください")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = state["cfg"]
    G = Generator(
        cfg["resolution"], cfg["z_channels"], cfg["latent_size"],
        cfg["base_channels"], cfg["max_channels"], cfg["n_res"], cfg["style_dim"],
    ).to(device)
    G.load_state_dict(state["G"])
    G.eval()

    tf = build_crop_transform(cfg["resolution"])
    amp = bool(cfg.get("amp")) and device == "cuda"

    xs, src_labels, tgt_labels = [], [], []
    for path, label in zip(args.images, args.labels):
        x = tf(Image.open(path)).unsqueeze(0).to(device)
        xs.append(x)
        src_labels.append(torch.tensor([label], device=device))
        tgt_labels.append(torch.tensor([1.0 - label], device=device))

    with torch.autocast("cuda", enabled=amp):
        fakes = [G(x, t) for x, t in zip(xs, tgt_labels)]

    n = len(xs)
    grid = (torch.cat(fakes) * 0.5 + 0.5).float().clamp(0, 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, out_path, nrow=n)
    print(f"saved {n} fake images -> {out_path}")


if __name__ == "__main__":
    main()
