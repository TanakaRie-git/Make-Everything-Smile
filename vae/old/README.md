# old/ — 変更前のスクラッチ VAE-GAN(アーカイブ)

diffusion トラックの**シャープさ**に手法を揃えるため、VAE トラックの実装を
「スクラッチ学習の VAE-GAN」から「**事前学習 SD-VAE を土台にした潜在編集**」へ
変更した。ここには**変更前**の一式を、当時の記録として凍結保存する。

## なぜ置き換えたか

- スクラッチ VAE-GAN は α=0(素の再構成)の時点で既にボケる。原因は手法族ではなく、
  **画像を描く decoder をゼロから小データ(512px・約9千枚)で学習**した点。
- diffusion がシャープなのは、笑顔を LoRA で少し足すだけで **画像↔潜在の autoencoder
  (SD-VAE)が事前学習済み・凍結**だから。
- そこで VAE 側も同じ土台(`stabilityai/sd-vae-ft-mse` 凍結)に載せ替えたのが現行本編
  [experiments/pretrained_ae/](../experiments/pretrained_ae/)。**α=0 の再構成が完全にシャープ**になり、
  物体を保ったまま笑顔が宿るようになった。

詳しい時系列(検証1〜9)は [docs/vae_verification_log.md](../docs/vae_verification_log.md)。
旧手法の代表画像は [assets/old/](../assets/old/)。

## 中身

| パス | 内容 |
|---|---|
| [smilevae/](smilevae/) | スクラッチ VAE-GAN の共有コア(models / train / losses / direction / edit)。`data.py` は現行の [src/smilevae/](../src/smilevae/) と共通のものを自己完結用に同梱 |
| [objects/](objects/) | 本編だった物体への笑顔付与(64/256/512px の config と手順) |
| [diffusion_compare/](diffusion_compare/) | diffusion トラックと同一 crop での比較(スクラッチ版) |

## 実行(参考・再現用)

現行の slim な `src/smilevae`(= data + viz のみ)とは別物なので、この凍結パッケージを
使うには `cd vae` から `PYTHONPATH=old` を付ける:

```bash
cd vae
PYTHONPATH=old uv run python -m smilevae.train old/objects/configs/smoke64.yaml
```
