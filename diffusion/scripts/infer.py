"""Inference: turn a face image into a smiling version using IP2P + trained LoRA.

使い方:
    uv run python scripts/infer.py \
        --lora_path outputs/smile_lora_v1/final/unet_lora \
        --input_image path/to/face.jpg \
        --output_dir outputs/results

引き継ぎ書のガイド:
  * image_guidance_scale（元画像保持の強さ）と guidance_scale（テキスト条件の強さ）は
    デフォルト 1.5 / 7.5 から。identity が崩れる場合は image_guidance_scale を 2.0〜2.5 に。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Smile inference with IP2P + LoRA")
    p.add_argument("--lora_path", required=True, type=str,
                   help="train.py が保存した LoRA ディレクトリ（pytorch_lora_weights.safetensors を含む）")
    p.add_argument("--input_image", required=True, type=str)
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
    return p.parse_args()


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

    img = Image.open(args.input_image).convert("RGB").resize((args.resolution, args.resolution))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    edited = pipe(
        args.prompt,
        image=img,
        num_inference_steps=args.num_inference_steps,
        image_guidance_scale=args.image_guidance_scale,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input_image).stem

    edited_path = out_dir / f"{stem}_smile.png"
    edited.save(edited_path)

    # 入力 | 出力 の比較画像も保存。
    combo = Image.new("RGB", (args.resolution * 2, args.resolution))
    combo.paste(img, (0, 0))
    combo.paste(edited, (args.resolution, 0))
    combo_path = out_dir / f"{stem}_compare.png"
    combo.save(combo_path)

    print(f"[done] {edited_path}")
    print(f"[done] {combo_path}")


if __name__ == "__main__":
    main()
