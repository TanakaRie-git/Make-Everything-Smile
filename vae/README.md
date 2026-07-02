# VAE トラック — 潜在編集による物体への笑顔付与

計画書([docs/make_everything_smile_plan.md](docs/make_everything_smile_plan.md))の VAE 担当分。
「顔ではない普通の物体」に、潜在空間の **笑顔方向ベクトル**(`z + α·direction`)を足して
笑った顔を宿す。これは DCGAN 由来の潜在ベクトル演算で、Diffusion トラックの Concept Slider と
同型の発想。

> 📋 検証の時系列(条件・結果・在処)は **[検証ログ](docs/vae_verification_log.md)** に、
> 発表用の説明は [docs/vae_presentation_content.md](docs/vae_presentation_content.md) にまとめている。

## 現行の手法(本編): 事前学習 SD-VAE を土台に

当初はスクラッチ学習の **VAE-GAN** を使ったが、再構成が常にボケた(→ [old/](old/))。
原因は手法族ではなく **画像を描く decoder をゼロから小データで学習**した点にある。
Diffusion がシャープなのは、笑顔を LoRA で少し足すだけで **画像↔潜在の autoencoder
(SD-VAE)が事前学習済み・凍結**だから。そこで VAE 側も同じ土台へ載せ替えた:

**事前学習 SD-VAE(`stabilityai/sd-vae-ft-mse`)を凍結し、その潜在で笑顔方向を操作する。**
→ α=0 の再構成が完全にシャープになり、物体を保ったまま笑顔が宿る。

手順・結果は **[experiments/pretrained_ae/README.md](experiments/pretrained_ae/README.md)**。

## セットアップ(uv 環境)

```bash
cd vae
uv sync          # PyTorch は CUDA 12.4 ホイール(pyproject.toml の index 設定)
```

## 構成

```
src/smilevae/       # 現行手法が使う共有コア(slim)
  data.py       # 白パディング正方形化 + リサイズ(FiT と作法統一、計画 §2.2)
  viz.py        # コンタクトシート描画ヘルパ(to_pil / make_sheet)
experiments/
  pretrained_ae/  # ★本編。凍結 SD-VAE の潜在で笑顔方向を操作(シャープ+物体保存)
  eval.py         # 軽量評価(LPIPS/SSIM/CLIP)
  eval_ids.txt    # 全手法共通の固定評価 ID(diffusion と同一 crop の 10 枚)
scripts/            # データ取得(CelebA / CelebA-HQ / Faces in Things / Tiny ImageNet)
docs/               # 計画書・検証ログ・発表用(テキスト)
assets/             # 掲載画像(docs から分離)。current/ old/ results/ に整理(下記)
old/                # 変更前のスクラッチ VAE-GAN 一式(アーカイブ、下記)
data/ outputs/      # gitignore 済み・再生成可能
```

## old/ — 変更前のスクラッチ VAE-GAN(アーカイブ)

diffusion に手法を揃える前の実装。スクラッチ VAE-GAN の学習コア・本編実験
(`objects/`)・diffusion 比較(`diffusion_compare/`)を凍結保存している。置き換えの
経緯と実行方法は **[old/README.md](old/README.md)**。当時の代表画像は
[assets/old/](assets/old/)。

## assets/ — 掲載画像

`docs/` はテキストのみとし、画像は `assets/` に分離して3つに整理した(索引 [assets/README.md](assets/README.md)):

| フォルダ | 中身 |
|---|---|
| [assets/current/](assets/current/) | 現行 SD-VAE 手法の代表作例・発表キービジュアル(`cmp_*`) |
| [assets/old/](assets/old/) | 変更前スクラッチ VAE-GAN の作例 |
| [assets/results/](assets/results/) | 全 ID overview シート + 評価メトリクス([results/README.md](assets/results/README.md)) |
