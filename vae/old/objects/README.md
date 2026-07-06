# 【アーカイブ】実験: 物体への笑顔付与(スクラッチ VAE-GAN)

> ⚠️ **これは変更前の旧手法です。** スクラッチ学習した VAE-GAN は再構成がボケる
> ため、現行の本編は事前学習 SD-VAE を土台にした [experiments/pretrained_ae/](../../experiments/pretrained_ae/)
> に置き換えた。経緯は [old/README.md](../README.md) を参照。以下は当時の記録として保存する。

VAE-GAN (Larsen et al. 2016) を CelebA(笑顔/中立)+ 物体画像で学習し、潜在空間の
**笑顔方向ベクトル**で物体画像を編集する。計画書 §0〜§3 の VAE 担当分。

このアーカイブ内の scratch パッケージ [old/smilevae/](../smilevae/) を使う
(`cd vae` から `PYTHONPATH=old` を付けて実行)。

## データ準備(スモークテスト用)

データは `vae/data/` に置く(gitignore 済み)。

```bash
# CelebA(aligned, 属性付き)から 笑顔/中立 × 男/女 を各1000枚(計4000枚)
uv run scripts/download_celeba.py --source celeba --n-per-class 1000 --out data/raw/celeba

# 物体: Tiny ImageNet を WordNet 階層で非生物クラスにフィルタ
uv run scripts/download_objects_smoke.py --out data/raw/objects_smoke --n-classes 30 --n-per-class 100

# (+Pareidolia 条件用) Faces in Things 公式 zip + happy サブセット抽出
uv run scripts/download_faces_in_things.py --out data/raw/faces_in_things

# (256px 本番用) CelebA-HQ。キャプションで笑顔判定
uv run scripts/download_celeba.py --source celeba-hq --n-per-class 5000 --out data/raw/celeba_hq
```

## 学習 → 笑顔方向 → 編集

```bash
# 1. VAE-GAN 学習(スモーク: 64px・4000 step・RTX 3070 Ti で十数分)
uv run python -m smilevae.train old/objects/configs/smoke64.yaml

# 2. 潜在空間の笑顔方向を算出(男女別ペアの平均で性別のもつれを軽減)
uv run python -m smilevae.direction \
  --ckpt outputs/smoke64_v3/ckpt/last.pt \
  --pair data/raw/celeba/smile_male data/raw/celeba/neutral_male \
  --pair data/raw/celeba/smile_female data/raw/celeba/neutral_female \
  --out outputs/smoke64_v3/smile_direction.pt

# 3. 物体画像へ弱/中/強の3段階で笑顔付与(計画 §0.3)
#    顔画像には α=1/2/3、物体には α=4/8/12 程度が目安(物体は顔多様体から
#    遠いため強めに押す必要がある。スモークv3の知見)
uv run python -m smilevae.edit \
  --ckpt outputs/smoke64_v3/ckpt/last.pt \
  --direction outputs/smoke64_v3/smile_direction.pt \
  --input-dir data/raw/objects_smoke/n02769748_backpack \
  --out-dir outputs/smoke64_v3/edits --limit 8 --strengths 4 8 12 --name backpack
```

出力はラベル付きコンタクトシート(列: `input | recon | weak | mid | strong`)。

## 2条件の作り分け(計画 §2.3)

config の `data_dirs` だけが違う:

- [configs/baseline256.yaml](configs/baseline256.yaml) — CelebA-HQ + 物体
- [configs/pareidolia256.yaml](configs/pareidolia256.yaml) — 上記 + Faces in Things (happy)

512px 版は [configs/hq512.yaml](configs/hq512.yaml) / [configs/pareidolia512.yaml](configs/pareidolia512.yaml)。

## スモークテストの知見(64px)

- グローバル潜在(v2)は笑顔編集は効くが、物体が顔そのものに置き換わる
  (物体保存ゼロ)。空間潜在 8×8(v3)で物体の構図・色が保持される。
- 笑顔方向は男女別に計算して平均(`Smiling`属性の性別もつれ対策)。
- 物体には顔より強い α が必要(顔 1〜3 / 物体 4〜12)。α を上げると
  「物体のまま」→「顔がうっすら宿る」→「顔が支配的」と遷移する。

## 既知の制約

- GPU は RTX 3070 Ti (8GB)。256px は batch 16 想定、必要なら勾配蓄積を足す。
- 本番の物体データ(ImageNet キュレーション)は横断担当の成果物を待つ。
  Tiny ImageNet はスモーク専用(64px しかない)。
- CelebA-HQ の笑顔ラベルはキャプション由来(`Ryan-sjtu/celebahq-caption`)。
  精度が気になる場合は CelebA 属性と突き合わせる。
