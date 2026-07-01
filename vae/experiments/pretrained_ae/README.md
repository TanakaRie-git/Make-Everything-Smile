# 実験: 事前学習オートエンコーダ(SD-VAE)を土台にした笑顔付与

スクラッチ学習の VAE-GAN([../objects/](../objects/) / [../diffusion_compare/](../diffusion_compare/))は
再構成がボケる。原因は手法族ではなく **画像を描く decoder をゼロから小データで学習**して
いるため。Diffusion トラックがシャープなのは、笑顔を LoRA で少し足すだけで **画像を描く
VAE(SD の autoencoder)は事前学習済み・凍結**だから(`diffusion/configs/train_lora.yaml`)。

本実験はその同じ土台を VAE トラック側でも使う: **事前学習済み SD-VAE(`stabilityai/
sd-vae-ft-mse`)を凍結**し、その潜在空間で笑顔方向ベクトルを操作する。手法は他の VAE 実験と
同じ `z = encode(x); decode(z + α·proj_std·direction)`、違いは encoder/decoder が事前学習
済みという点だけ。

## 手順(cd vae から)

```bash
# 1. SD-VAE 潜在で笑顔方向を算出(男女別ペア平均。初回は SD-VAE 重みを自動DL)
uv run python experiments/pretrained_ae/build_direction.py \
  --pair data/raw/celeba_hq/smile_male   data/raw/celeba_hq/neutral_male \
  --pair data/raw/celeba_hq/smile_female data/raw/celeba_hq/neutral_female \
  --out outputs/pretrained_ae/smile_direction.pt \
  --resolution 512 --batch-size 8 --limit-per-folder 500

# 2. diffusion_compare と同一の crop・同一プロトコルで編集
uv run python experiments/pretrained_ae/compare_facesinthings.py \
  --direction outputs/pretrained_ae/smile_direction.pt \
  --input-dir data/diffusion_compare/inputs \
  --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/pretrained_ae/compare --scales 0 1 2 3
```

出力形式は他実験と揃える: `<id>_compare.png`(input | α=0 | …)・`<id>_smile.png`・`_overview.png`。

## 構成

- `sdvae.py` — 事前学習 `AutoencoderKL` の load / encode(潜在の平均)/ decode ラッパ(凍結)。
- `build_direction.py` — CelebA-HQ ペアを SD-VAE で encode し笑顔方向を算出(`smilevae.direction` と同設計)。
- `compare_facesinthings.py` — 評価 crop を encode → scale ラダーで decode。描画は共有の `smilevae.edit` を再利用。

## 結果(初回ラン、共通10 ID・512px)

- **α=0(再構成)が完全にシャープ**。物体のテクスチャ・輪郭・色を忠実に保持
  (スクラッチ版は α=0 で既にボケていた)。
- **物体を保ったまま、笑った口(歯)が局所的に宿る**。ワッフル/蒸しパン/木目などが
  それぞれの見た目のまま笑顔化。全体が人間に置換されない(diffusion_compare のスクラッチ版とは対照的)。
- α を上げる(2〜3)と目・口の顔構造が強まる。0〜3 が使いやすい(proj_std≈118)。

## α とスケールの目安

α は proj_std 単位。SD-VAE 潜在は proj_std が大きい(≈118)ので、`0 1 2 3` 程度で十分効く。
スクラッチ版(`0 4 8 12`)とは効く α の範囲が違う点に注意。

## 依存

`diffusers` / `safetensors`(共有 `pyproject.toml` に追加済み)。SD-VAE 重み ~335MB を
HuggingFace から自動DL(初回のみ、ネットワーク必要)。
