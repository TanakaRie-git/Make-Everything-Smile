"""Inference: turn a face image into a smiling version using IP2P + trained LoRA.

入力は 2 通り:

(A) 単一画像（従来どおり）:
    uv run python scripts/infer.py \
        --lora_path outputs/smile_lora_v1/final/unet_lora \
        --input_image path/to/face.jpg \
        --output_dir outputs/results

(B) FacesInThings データセット（パレイドリア＝物体に見える顔）を一括入力:
    uv run python scripts/infer.py \
        --lora_path outputs/smile_lora_v1/final/unet_lora \
        --facesinthings_root ../data --download \
        --max_images 20 --use_box \
        --output_dir outputs/pareidolia

  * FacesInThings は「物の中の顔」を矩形(boxes)付きで持つ。--use_box を付けると
    顔矩形にクロップしてから笑顔化する（パレイドリアの顔は画像の一部なので、
    画像全体を通すより顔領域を切り出した方が効きやすい）。--paste_back で
    編集後のクロップを元画像に貼り戻した合成も保存する。

引き継ぎ書のガイド:
  * image_guidance_scale（元画像保持の強さ）と guidance_scale（テキスト条件の強さ）は
    デフォルト 1.5 / 7.5 から。identity が崩れる場合は image_guidance_scale を 2.0〜2.5 に。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

# scripts/ から data/ パッケージを import できるようにリポジトリルートを通す。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Smile inference with IP2P + LoRA")
    p.add_argument("--lora_path", type=str, default=None,
                   help="train.py が保存した LoRA ディレクトリ（pytorch_lora_weights.safetensors を含む）")
    # --- 入力ソース（どちらか一方） ---
    p.add_argument("--input_image", type=str, default=None,
                   help="単一画像を入力する場合のパス")
    p.add_argument("--facesinthings_root", type=str, default=None,
                   help="FacesInThings の親ディレクトリ（<root>/FacesInThings/ を探す）")
    # --- FacesInThings 用オプション ---
    p.add_argument("--download", action="store_true",
                   help="FacesInThings が無ければ https://aka.ms/faces-dataset からDL")
    p.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    p.add_argument("--primary_only", action="store_true",
                   help="is_primary な画像だけを対象にする")
    p.add_argument("--max_images", type=int, default=None,
                   help="処理する最大枚数（動作確認用に絞る）")
    p.add_argument("--use_box", action="store_true",
                   help="顔矩形(boxes)にクロップしてから笑顔化する")
    p.add_argument("--box_pad", type=float, default=0.3,
                   help="--use_box 時、矩形の周囲に足す余白（矩形サイズに対する割合）")
    p.add_argument("--paste_back", action="store_true",
                   help="--use_box 時、編集後クロップを元画像に貼り戻した合成も保存")
    # --- 共通 ---
    p.add_argument("--output_dir", type=str, default="outputs/results")
    p.add_argument("--base_model", type=str, default="timbrooks/instruct-pix2pix")
    p.add_argument("--prompt", type=str, default="make the person smile")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--image_guidance_scale", type=float, default=1.5)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_lora", action="store_true",
                   help="LoRA を読まず素の IP2P で推論（比較用）")
    args = p.parse_args()

    if not args.input_image and not args.facesinthings_root:
        p.error("--input_image か --facesinthings_root のどちらかを指定してください。")
    if args.input_image and args.facesinthings_root:
        p.error("--input_image と --facesinthings_root は同時に指定できません。")
    if not args.no_lora and not args.lora_path:
        p.error("--lora_path を指定するか、--no_lora を付けてください。")
    return args


def crop_box(img: Image.Image, box: list[float], pad: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """[x1, y1, w, h] の顔矩形を pad 割合ぶん広げて正方形でクロップする。

    返り値: (クロップ画像, 実際に切り出した (left, top, right, bottom))。
    貼り戻し用に切り出し座標も返す。
    """
    x1, y1, w, h = box
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    side = max(w, h) * (1.0 + pad)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = int(round(cx + side / 2.0))
    bottom = int(round(cy + side / 2.0))
    # 画像内にクランプ。
    left = max(0, left)
    top = max(0, top)
    right = min(img.width, right)
    bottom = min(img.height, bottom)
    return img.crop((left, top, right, bottom)), (left, top, right, bottom)


def edit_image(pipe, img: Image.Image, args, generator) -> Image.Image:
    """1 枚の PIL 画像を笑顔化して返す（resolution に合わせて in/out する）。"""
    work = img.resize((args.resolution, args.resolution))
    edited = pipe(
        args.prompt,
        image=work,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    return edited


def save_compare(original: Image.Image, edited: Image.Image, path: Path, resolution: int):
    """入力 | 出力 の横並び比較画像を保存。"""
    left = original.resize((resolution, resolution))
    right = edited.resize((resolution, resolution))
    combo = Image.new("RGB", (resolution * 2, resolution))
    combo.paste(left, (0, 0))
    combo.paste(right, (resolution, 0))
    combo.save(path)


def run_single_image(pipe, args, generator, out_dir: Path):
    img = Image.open(args.input_image).convert("RGB")
    edited = edit_image(pipe, img, args, generator)
    stem = Path(args.input_image).stem

    edited_path = out_dir / f"{stem}_smile.png"
    edited.save(edited_path)
    combo_path = out_dir / f"{stem}_compare.png"
    save_compare(img, edited, combo_path, args.resolution)

    print(f"[done] {edited_path}")
    print(f"[done] {combo_path}")


def run_facesinthings(pipe, args, generator, out_dir: Path):
    from data.facesinthings import FacesInThings

    ds = FacesInThings(
        root=args.facesinthings_root,
        download=args.download,
        split=args.split,
        primary_only=args.primary_only,
    )
    n = len(ds) if args.max_images is None else min(len(ds), args.max_images)
    print(f"[facesinthings] {len(ds)} 枚中 {n} 枚を処理します (use_box={args.use_box})")

    for i in range(n):
        sample = ds[i]
        try:
            full = sample.open_image()
        except FileNotFoundError:
            print(f"[skip] 画像が見つかりません: {sample.path}")
            continue

        stem = Path(sample.file).stem
        box = sample.primary_box() if args.use_box else None

        if box is not None:
            crop, (left, top, right, bottom) = crop_box(full, box, args.box_pad)
            edited_crop = edit_image(pipe, crop, args, generator)

            edited_crop.save(out_dir / f"{stem}_smile.png")
            save_compare(crop, edited_crop, out_dir / f"{stem}_compare.png", args.resolution)

            if args.paste_back:
                composite = full.copy()
                # クロップは正方形・任意サイズなので、切り出した実寸に戻して貼る。
                edited_resized = edited_crop.resize((right - left, bottom - top))
                composite.paste(edited_resized, (left, top))
                composite.save(out_dir / f"{stem}_pasteback.png")
        else:
            # 矩形を使わず画像全体を笑顔化（--use_box 無し、または box が無い画像）。
            edited = edit_image(pipe, full, args, generator)
            edited.save(out_dir / f"{stem}_smile.png")
            save_compare(full, edited, out_dir / f"{stem}_compare.png", args.resolution)

        print(f"[{i + 1}/{n}] {sample.file}")

    print(f"[done] 出力: {out_dir}")


def main():
    from diffusers import StableDiffusionInstructPix2PixPipeline

    args = parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.base_model, torch_dtype=dtype, safety_checker=None
    )

    if not args.no_lora:
        lora_path = Path(args.lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA が見つかりません: {lora_path}")
        # ディレクトリでもファイルでも load_lora_weights が解決する。
        pipe.load_lora_weights(str(lora_path))
        print(f"[info] loaded LoRA: {lora_path}")

    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.facesinthings_root:
        run_facesinthings(pipe, args, generator, out_dir)
    else:
        run_single_image(pipe, args, generator, out_dir)


if __name__ == "__main__":
    main()
