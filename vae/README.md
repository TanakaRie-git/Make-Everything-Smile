# VAE トラック — VAE-GAN による物体への笑顔付与

計画書([docs/make_everything_smile_plan.md](docs/make_everything_smile_plan.md))の
VAE 担当分。VAE-GAN (Larsen et al. 2016) を CelebA(笑顔/中立)で学習し、潜在空間の
**笑顔方向ベクトル**(`z + α·direction`)で画像を編集する。これは DCGAN 由来の潜在
ベクトル演算であり、Diffusion トラックの Concept Slider と同型の発想。

> 📋 これまでの検証(条件・結果・在処)は **[検証ログ](docs/vae_verification_log.md)** にまとめている。

## セットアップ(共有 uv 環境)

```bash
cd vae
uv sync          # PyTorch は CUDA 12.4 ホイール (pyproject.toml の index 設定)
```

学習・編集の共有コアは `src/smilevae`(下記)。両実験はこの同じ環境・同じ学習済み
モデルを使う。

## 2つの実験

| 実験 | 中身 | 手順 |
|---|---|---|
| **[experiments/objects/](experiments/objects/)** | 本編。物体(Tiny ImageNet / ImageNet 系)へ弱/中/強で笑顔を付与し、コンタクトシートで比較 | [experiments/objects/README.md](experiments/objects/README.md) |
| **[experiments/diffusion_compare/](experiments/diffusion_compare/)** | Diffusion トラックと**同一の FacesInThings crop・同一プロトコル**(scale ラダー)で笑顔付与を再現し、手法間比較する | [experiments/diffusion_compare/README.md](experiments/diffusion_compare/README.md) |

## 共有コアの構成

```
src/smilevae/       # 両実験が共有するモデル/学習/編集コア
  models.py     # Encoder / Decoder / Discriminator(解像度可変)
                #   latent_size=0: グローバル潜在ベクトル(物体が顔に崩壊しやすい)
                #   latent_size≥2: 空間潜在マップ(物体の構図を保持。既定)
  data.py       # 白パディング正方形化 + リサイズ(FiT と作法統一、計画 §2.2)
  train.py      # VAE-GAN 学習(D: real/recon/prior、E+G: KL+特徴量再構成+adv)
  direction.py  # 笑顔方向 = normalize(平均_pair(mean(mu_smile) - mean(mu_neutral)))
  edit.py       # z + α·proj_std·direction をデコード(α = 弱/中/強)
scripts/            # データ取得(CelebA / CelebA-HQ / Faces in Things / Tiny ImageNet)
experiments/        # 実験ごとの config・手順・出力(上表)
docs/               # 計画書・検証ログ
data/ outputs/      # gitignore 済み・再生成可能
```
