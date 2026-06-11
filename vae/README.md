# VAE トラック — VAE-GAN による物体への笑顔付与

計画書([docs/make_everything_smile_plan.md](../docs/make_everything_smile_plan.md))の
VAE 担当分。VAE-GAN (Larsen et al. 2016) を CelebA(笑顔/中立)+物体画像で学習し、
潜在空間の**笑顔方向ベクトル**で物体画像を編集する。

## セットアップ

```bash
cd vae
uv sync          # PyTorch は CUDA 12.4 ホイール (pyproject.toml の index 設定)
```

## データ準備(スモークテスト用)

データはリポジトリ直下の共有 `data/` に置く(gitignore 済み)。

```bash
# CelebA(aligned, 属性付き)から笑顔/中立を各2000枚
uv run scripts/download_celeba.py --source celeba --n-per-class 2000 --out ../data/raw/celeba

# 物体: Tiny ImageNet を WordNet 階層で非生物クラスにフィルタ
uv run scripts/download_objects_smoke.py --out ../data/raw/objects_smoke --n-classes 30 --n-per-class 100

# (+Pareidolia 条件用) Faces in Things 公式 zip + happy サブセット抽出
uv run scripts/download_faces_in_things.py --out ../data/raw/faces_in_things

# (256px 本番用) CelebA-HQ。キャプションで笑顔判定
uv run scripts/download_celeba.py --source celeba-hq --n-per-class 5000 --out ../data/raw/celeba_hq
```

## 学習 → 笑顔方向 → 編集

```bash
# 1. VAE-GAN 学習(スモーク: 64px・4000 step・RTX 3070 Ti で十数分)
uv run python -m smilevae.train configs/smoke64.yaml

# 2. 潜在空間の笑顔方向ベクトルを算出
uv run python -m smilevae.direction \
  --ckpt outputs/smoke64/ckpt/last.pt \
  --smile-dir ../data/raw/celeba/smile --neutral-dir ../data/raw/celeba/neutral \
  --out outputs/smoke64/smile_direction.pt

# 3. 物体画像へ弱/中/強の3段階で笑顔付与(計画 §0.3)
uv run python -m smilevae.edit \
  --ckpt outputs/smoke64/ckpt/last.pt \
  --direction outputs/smoke64/smile_direction.pt \
  --input-dir ../data/raw/objects_smoke --out-dir outputs/smoke64/edits --limit 32
```

出力列: `input | recon | weak | mid | strong`(`_contact_sheet.png` に一覧)。

## 2条件の作り分け(計画 §2.3)

config の `data_dirs` だけが違う:

- [configs/baseline256.yaml](configs/baseline256.yaml) — CelebA-HQ + 物体
- [configs/pareidolia256.yaml](configs/pareidolia256.yaml) — 上記 + Faces in Things (happy)

## 構成

```
src/smilevae/
  models.py     # Encoder / Decoder / Discriminator(解像度可変、4x4まで縮小)
  data.py       # 白パディング正方形化 + リサイズ(FiT と作法統一、計画 §2.2)
  train.py      # VAE-GAN 学習(D: real/recon/prior、E+G: KL+特徴量再構成+adv)
  direction.py  # 笑顔方向 = normalize(mean(mu_smile) - mean(mu_neutral))
  edit.py       # z + α·proj_std·direction をデコード(α = 弱/中/強)
scripts/        # データ取得(CelebA / CelebA-HQ / Faces in Things / Tiny ImageNet)
configs/        # smoke64 / baseline256 / pareidolia256
```

## 既知の制約

- GPU は RTX 3070 Ti (8GB)。256px は batch 16 想定、必要なら勾配蓄積を足す。
- 本番の物体データ(ImageNet キュレーション)は横断担当の成果物を待つ。
  Tiny ImageNet はスモーク専用(64px しかない)。
- CelebA-HQ の笑顔ラベルはキャプション由来(`Ryan-sjtu/celebahq-caption`)。
  精度が気になる場合は CelebA 属性と突き合わせる。
