"""InstructPix2Pix + LoRA fine-tuning for non-smiling -> smiling face transfer.

引き継ぎ書の重要ポイントを実装している:
  * U-Net 入力は 8ch = concat([noisy_target_latent(4ch), original_image_latent(4ch)])。
    base model に IP2P (`timbrooks/instruct-pix2pix`) を使うので unet.in_channels==8。
  * LoRA は U-Net の attention 層 (to_k/to_q/to_v/to_out.0) のみに peft 経由で適用。
    VAE / text encoder / U-Net 本体は凍結。
  * まず resolution=256, batch=4 で動作確認する。

使い方:
    uv run python scripts/train.py --config configs/train_lora.yaml

学習済み LoRA は <output_dir>/final/unet_lora/ に保存される（infer.py がこれを読む）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# data パッケージを import 可能にする（リポジトリルートを sys.path に追加）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import SmilePairDataset, collate_fn  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train IP2P + LoRA for smile transfer")
    parser.add_argument("--config", required=True, type=str)
    # CLI で上書きしたい時用（任意）。例: --set resolution=512 train_batch_size=2
    parser.add_argument("--set", nargs="*", default=[],
                        help="key=value で config を上書き")
    return parser.parse_args()


def load_config(args):
    cfg = OmegaConf.load(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))
    return cfg


def tokenize_prompts(tokenizer, prompts):
    return tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids


def main(cfg):
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import (
        AutoencoderKL,
        DDPMScheduler,
        StableDiffusionInstructPix2PixPipeline,
        UNet2DConditionModel,
    )
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    output_dir = Path(cfg.output_dir)
    logging_dir = output_dir / cfg.get("logging_dir", "logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with="tensorboard",
        project_dir=str(logging_dir),
    )

    if cfg.get("seed") is not None:
        set_seed(cfg.seed)

    # --- モデル読み込み ---
    model_id = cfg.pretrained_model
    revision = cfg.get("revision")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", revision=revision)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # IP2P であることのサニティチェック（8ch 入力）。
    if unet.config.in_channels != 8:
        raise ValueError(
            f"U-Net in_channels={unet.config.in_channels} (expected 8). "
            f"IP2P 系のモデルを base にしてください（現在: {model_id}）。"
        )

    # base を凍結。
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # --- LoRA を attention 層に適用 ---
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.get("lora_dropout", 0.0),
        init_lora_weights="gaussian",
        target_modules=list(cfg.lora_target_modules),
    )
    unet.add_adapter(lora_config)

    # 混合精度: base の重みは weight_dtype、LoRA は fp32 で学習。
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # 学習対象 = LoRA パラメータのみ。fp16 だと不安定なので fp32 にキャストする。
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    if accelerator.mixed_precision == "fp16":
        for p in lora_params:
            p.data = p.data.float()
    n_trainable = sum(p.numel() for p in lora_params)
    accelerator.print(f"[info] trainable LoRA params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(lora_params, lr=cfg.learning_rate)

    # --- データ ---
    train_ds = SmilePairDataset(cfg.train_pairs, resolution=cfg.resolution,
                                random_flip=cfg.get("random_flip", True))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.get("dataloader_num_workers", 4),
        drop_last=True,
    )

    # --- step 数の決定 ---
    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if cfg.get("max_train_steps"):
        max_train_steps = int(cfg.max_train_steps)
        num_train_epochs = math.ceil(max_train_steps / steps_per_epoch)
    else:
        num_train_epochs = int(cfg.num_train_epochs)
        max_train_steps = steps_per_epoch * num_train_epochs

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * cfg.gradient_accumulation_steps,
        num_training_steps=max_train_steps * cfg.gradient_accumulation_steps,
    )

    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("smile_lora", config=OmegaConf.to_container(cfg, resolve=True))

    # null conditioning（CFG 用ドロップアウト時のテキスト埋め込み）を準備。
    null_ids = tokenize_prompts(tokenizer, [""]).to(accelerator.device)
    with torch.no_grad():
        null_embeds = text_encoder(null_ids)[0]

    cond_drop = cfg.get("conditioning_dropout_prob", 0.0)
    vae_scale = vae.config.scaling_factor

    accelerator.print(
        f"[info] epochs={num_train_epochs} steps/epoch={steps_per_epoch} "
        f"max_steps={max_train_steps} res={cfg.resolution} bs={cfg.train_batch_size}"
    )

    global_step = 0
    for epoch in range(num_train_epochs):
        unet.train()
        for batch in train_loader:
            with accelerator.accumulate(unet):
                target_px = batch["target_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                orig_px = batch["original_pixel_values"].to(accelerator.device, dtype=weight_dtype)

                # 教師(笑顔)の latent（scaling factor をかける）。
                with torch.no_grad():
                    latents = vae.encode(target_px).latent_dist.sample() * vae_scale
                    # 入力(非笑顔)の画像 latent（IP2P は scaling factor をかけず mode を使う）。
                    orig_latents = vae.encode(orig_px).latent_dist.mode()

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # テキスト条件。
                input_ids = tokenize_prompts(tokenizer, batch["instruction"]).to(accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # --- conditioning dropout（IP2P の CFG 学習） ---
                if cond_drop and cond_drop > 0.0:
                    random_p = torch.rand(bsz, device=latents.device)
                    # テキスト条件を null に落とす。
                    prompt_mask = (random_p < 2 * cond_drop).reshape(bsz, 1, 1)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_embeds.to(encoder_hidden_states.dtype), encoder_hidden_states
                    )
                    # 画像条件を 0 に落とす。
                    image_mask = 1.0 - (
                        (random_p >= cond_drop).float() * (random_p < 3 * cond_drop).float()
                    )
                    orig_latents = image_mask.reshape(bsz, 1, 1, 1) * orig_latents

                # 8ch concat。
                model_input = torch.cat([noisy_latents, orig_latents], dim=1)

                model_pred = unet(model_input, timesteps, encoder_hidden_states).sample

                # IP2P / SD は epsilon 予測。
                if noise_scheduler.config.prediction_type == "v_prediction":
                    gt = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    gt = noise
                loss = F.mse_loss(model_pred.float(), gt.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.detach().item(),
                                     "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    if global_step % 20 == 0:
                        accelerator.print(f"step {global_step}/{max_train_steps} loss={loss.item():.4f}")

                    if cfg.get("checkpointing_steps") and global_step % cfg.checkpointing_steps == 0:
                        save_lora(accelerator, unet, output_dir / f"checkpoint-{global_step}" / "unet_lora",
                                  get_peft_model_state_dict, StableDiffusionInstructPix2PixPipeline)

                if (cfg.get("validation_steps") and global_step % cfg.validation_steps == 0
                        and cfg.get("val_pairs")):
                    run_validation(accelerator, cfg, model_id, revision, unet, vae, text_encoder,
                                   tokenizer, weight_dtype, global_step)
                    unet.train()

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    # --- 最終保存 ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(accelerator, unet, output_dir / "final" / "unet_lora",
                  get_peft_model_state_dict, StableDiffusionInstructPix2PixPipeline)
        accelerator.print(f"[done] LoRA saved -> {output_dir / 'final' / 'unet_lora'}")
    accelerator.end_training()


def save_lora(accelerator, unet, save_dir, get_peft_model_state_dict, pipe_cls):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(unet)
    lora_state = get_peft_model_state_dict(unwrapped)
    # diffusers 形式 (pytorch_lora_weights.safetensors) で保存 → infer.py の load_lora_weights が読む。
    pipe_cls.save_lora_weights(save_directory=str(save_dir), unet_lora_layers=lora_state, safe_serialization=True)
    accelerator.print(f"[ckpt] {save_dir}")


@torch.no_grad()
def run_validation(accelerator, cfg, model_id, revision, unet, vae, text_encoder,
                   tokenizer, weight_dtype, step):
    """検証画像を生成して TensorBoard / ディスクに保存する。"""
    import json

    from diffusers import StableDiffusionInstructPix2PixPipeline
    from PIL import Image

    accelerator.print(f"[val] running validation at step {step}")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=revision,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(accelerator.device)

    val_pairs = json.loads(Path(cfg.val_pairs).read_text(encoding="utf-8"))
    n = min(cfg.get("num_validation_images", 4), len(val_pairs))

    out_dir = Path(cfg.output_dir) / "validation" / f"step-{step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(cfg.get("seed", 0) or 0)
    for i in range(n):
        item = val_pairs[i]
        src = Image.open(item["original"]).convert("RGB").resize((cfg.resolution, cfg.resolution))
        edited = pipe(
            item.get("instruction", "make the person smile"),
            image=src,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]
        # 入力 | 出力 を横並びにして保存。
        combo = Image.new("RGB", (cfg.resolution * 2, cfg.resolution))
        combo.paste(src, (0, 0))
        combo.paste(edited, (cfg.resolution, 0))
        combo.save(out_dir / f"val_{i:02d}.png")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.print(f"[val] saved -> {out_dir}")


def main_cli():
    args = parse_args()
    cfg = load_config(args)
    main(cfg)


if __name__ == "__main__":
    main_cli()
