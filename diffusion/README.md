# Smile Diffusion (InstructPix2Pix + LoRA)

非笑顔の顔画像を笑顔に変換するモデルを、**InstructPix2Pix (IP2P) + LoRA fine-tuning**
で実装する。データセットは MM-CelebA-HQ。

> `~/work/Make-Everything-Smile/` の `diffusion/` 担当領域。`vae/`, `gan/` は別担当のため触らない。
> uv 環境は diffusion 単体で独立（torch のバージョンが他チームと異なる）。

## ディレクトリ構成

```
diffusion/
├── pyproject.toml          # 依存関係（torch / diffusers / peft 等, CUDA 12.8 ビルド）
├── configs/
│   └── train_lora.yaml     # 学習設定
├── data/
│   ├── build_pairs.py      # 非笑顔→笑顔ペア構築（属性ベースのランダムサンプリング）
│   ├── dataset.py          # PyTorch Dataset
│   └── facesinthings.py    # FacesInThings（物に見える顔）ローダ（推論時の入力に使用）
├── scripts/
│   ├── train.py            # 学習メイン（IP2P 8ch 入力 + peft LoRA）
│   └── infer.py            # 推論（単一画像 / FacesInThings 一括）
└── README.md
```

## セットアップ

```bash
cd ~/work/Make-Everything-Smile/diffusion
uv sync                      # RTX 5080 / CUDA 12.8 用 torch が入る
```

## 1. ペアデータ構築

CelebA-HQ は次を想定（パスは実配置に合わせる）:

```
/path/to/celeba_hq/
├── CelebA-HQ-img/                       # 画像
└── CelebAMask-HQ-attribute_list.txt     # 属性（Smiling 列を使用）
```

```bash
uv run python data/build_pairs.py \
  --data_root /path/to/celeba_hq \
  --out_dir pairs \
  --max_pairs 10000 \
  --val_ratio 0.1
```

→ `pairs/train.json`, `pairs/val.json` が生成される。

**設計判断**: CelebA-HQ には同一人物の笑顔/非笑顔ペアが存在しないため、
`Smiling=0` と `Smiling=1` を**別々にランダムサンプリング**してペアにしている。
品質が低い場合は identity ベースのペアリングに拡張する（コード内に TODO）。

## 2. 学習

```bash
uv run python scripts/train.py --config configs/train_lora.yaml
```

ポイント（`configs/train_lora.yaml` で調整）:
- **解像度**: まず `resolution: 256`, `train_batch_size: 4` で動作確認 → 通ってから 512px。
- **IP2P 8ch 入力**: U-Net への入力は `[noisy_target_latent(4ch), original_image_latent(4ch)]`
  を concat した 8ch。base model は `timbrooks/instruct-pix2pix` 必須（SD1.5 だと shape mismatch）。
- **LoRA**: U-Net の attention 層 (`to_k/to_q/to_v/to_out.0`) のみに適用。VAE / text encoder は凍結。

成果物:
- LoRA: `outputs/smile_lora_v1/final/unet_lora/`（中間: `checkpoint-<step>/unet_lora/`）
- 検証画像: `outputs/smile_lora_v1/validation/step-<n>/`
- TensorBoard: `outputs/smile_lora_v1/logs/`

```bash
uv run tensorboard --logdir outputs/smile_lora_v1/logs
```

CLI で一時上書きも可能:

```bash
uv run python scripts/train.py --config configs/train_lora.yaml \
  --set resolution=512 train_batch_size=2 max_train_steps=2000
```

## 3. 推論

入力は 2 通り。

### (A) 単一画像

```bash
uv run python scripts/infer.py \
  --lora_path outputs/smile_lora_v1/final/unet_lora \
  --input_image path/to/face.jpg \
  --output_dir outputs/results
```

→ `<name>_smile.png`（変換結果）と `<name>_compare.png`（入力|出力 比較）を出力。

### (B) FacesInThings 一括（パレイドリア＝物に見える顔）

`data/facesinthings.py` が読む FacesInThings データセット（物の中の顔を矩形付きで持つ）を
入力に、まとめて笑顔化する。

```bash
uv run python scripts/infer.py \
  --lora_path outputs/smile_lora_v1/final/unet_lora \
  --facesinthings_root ../data --download \
  --max_images 20 --use_box \
  --output_dir outputs/pareidolia
```

- `--facesinthings_root`: `<root>/FacesInThings/` を探す。`--download` で無ければ
  `https://aka.ms/faces-dataset` から取得・展開。
- `--use_box`: 顔矩形(boxes)にクロップしてから笑顔化（画像全体より顔領域を切り出した方が効きやすい）。
  `--box_pad`（既定 0.3）で矩形周囲の余白、`--paste_back` で編集後クロップを元画像に貼り戻した合成も保存。
- `--split {all,train,test}` / `--primary_only` / `--max_images` で対象を絞り込む。

### 共通の調整

`--image_guidance_scale`（元画像保持, 既定 1.5）と `--guidance_scale`
（テキスト条件, 既定 7.5）。identity が崩れる場合は `image_guidance_scale` を 2.0〜2.5 に上げる。

素の IP2P と比較したい場合は `--no_lora` を付ける。

## TODO / 将来拡張

- identity ベースのペアリング（`build_pairs.py`）— 品質改善時に検討。
- LPIPS 損失の追加（`train.py` は現状 epsilon-MSE のみ）。
- 512px での VRAM 使用量・学習時間の記録。
