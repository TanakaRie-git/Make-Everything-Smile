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

## 笑顔方向の選択(何から「笑顔」を学ぶか) — Diffusion との整合

Diffusion の pareidria policy は **FacesInThings 自体の neutral→happy** を学習し、IP2P の
画像条件つき編集で「物体の顔そのものを笑わせる」(`diffusion/data/build_pairs_pareidria.py`,
`diffusion/scripts/infer.py`)。VAE 側も**同じドメインから方向を学ぶ**と挙動が揃う:

| 方向ファイル | 学習ソース | 見え方 |
|---|---|---|
| `smile_direction.pt` | **CelebA 人間の笑顔** | 人の顔・歯が"重なる"(overlay 的) |
| `fit_direction.pt` | **FacesInThings neutral→happy**(diffusion と同ドメイン) | **物体のまま**表情が happy 化 |

`fit_direction` の作り方:
```bash
uv run python experiments/pretrained_ae/build_direction.py \
  --pair data/facesinthings_crops/happy data/facesinthings_crops/neutral \
  --out outputs/pretrained_ae/fit_direction.pt --resolution 512 --limit-per-folder 800
# 編集: --direction outputs/pretrained_ae/fit_direction.pt --scales 0 1 2 3
```
crop は diffusion と同一(`diffusion/data/cache/facesinthings_crops/` を `data/facesinthings_crops/` にコピー)。

## α とスケールの目安・既知の差

- α は proj_std 単位。`smile_direction`(proj_std≈118)は `0 1 2 3`、`fit_direction`
  (proj_std≈214)は `0 1 2 3` で暖色ドリフトが強く出るので小さめ推奨。
- スクラッチ版(`0 4 8 12`)とは効く α の範囲が違う。
- **残る差**: VAE は潜在に一律方向を足すため、`fit_direction` では α 増で**全体が暖色に寄る
  色ドリフト**が出る(happy 群の色偏り)。Diffusion は画像条件つきで空間的に必要箇所だけ
  編集するため色ドリフトが出ない。「グローバル方向 vs 画像条件つき編集」の本質差。

## 色ドリフトの抑制(笑顔だけ残す)

`compare_facesinthings.py` に2つの抑制オプションを用意:

| オプション | 効果 |
|---|---|
| `--feature-only` | 方向の空間平均(=全体の色/トーンを一律に押す成分)を除去。局所的な表情構造だけ残す。除去分だけ効きが弱まるので α を上げる(例 `0 2 4 6`)。 |
| `--keep-color` | decode 後に **輝度は編集後・色相/彩度は入力**へ戻す(YCbCr合成)。色ドリフトをほぼ完全に打ち消す。**推奨**。 |

```bash
# 推奨: 色を保ったまま笑顔だけ乗せる
uv run python experiments/pretrained_ae/compare_facesinthings.py \
  --direction outputs/pretrained_ae/fit_direction.pt \
  --input-dir data/diffusion_compare/inputs --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/pretrained_ae/compare_fit_kc --scales 0 1 2 3 --keep-color
```

**結果**: `--keep-color`(または `--feature-only --keep-color`)で色ドリフトが消え、
**物体は元の色・形のまま、自分の口が笑みのカーブになる**(ワッフル/ポテト/蒸しパン等で
口角が上がる)。「物体自身の顔を笑わせる」= Diffusion のタスク定義に最も近い出力。
原本: `outputs/pretrained_ae/compare_fit_kc/`(keep-color)/ `compare_fit_both/`(両方)。

## 2段カスケード(顔っぽくしてから人間の笑顔ポリシー)

「一度顔っぽくした画像を、改めて人間の笑顔ポリシー(CelebA 方向)に通せば、もう顔に
近いので笑顔が作りやすいのでは」という仮説の検証。`cascade_smile.py` が
**生成画像を decode→再encode して次段に渡す**真のカスケードを行う:
```
face = decode(encode(x) + α1·d1)   # Stage1: 顔っぽい画像
out  = decode(encode(face) + α2·d2) # Stage2: その画像に人間の笑顔方向を α ラダーで
```
```bash
uv run python experiments/pretrained_ae/cascade_smile.py \
  --stage1-direction outputs/pretrained_ae/smile_direction.pt --stage1-alpha 2 \
  --stage2-direction outputs/pretrained_ae/smile_direction.pt --stage2-scales 0 1 2 3 \
  --input-dir data/diffusion_compare/inputs --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/pretrained_ae/cascade
```
**結果**: 仮説どおり、顔化後に笑顔方向を足すと **α=2〜3 で歯を見せた明確な笑顔**になり、
素の物体を直接笑わせるより笑顔が強く・くっきり出る。ただし**物体は人間の顔になりきる**
(物体保存は失う)。→ `fit_direction`+`--keep-color`(物体を保ち控えめに笑う)と対の関係で、
**「物体を残す」↔「笑顔を強くする」の2極**を成す。原本 `outputs/pretrained_ae/cascade/`。

## 個性を保つ低α版(「みんな同じ顔」を避ける)

高 α で全物体が同じ平均顔に収束するのは、**一律ベクトル `α·d` が入力固有の `z` を圧倒**する
ため(=強い笑顔と個体保存のトレードオフ)。α を下げれば `z`(物体の個性)が残る。

```bash
# 推奨: CelebA 方向・低α・色保持(各物体が別物のまま口元に笑み)
uv run python experiments/pretrained_ae/compare_facesinthings.py \
  --direction outputs/pretrained_ae/smile_direction.pt --scales 0 0.5 1 1.5 --keep-color \
  --input-dir data/diffusion_compare/inputs --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/pretrained_ae/low_celeba_kc
```

生成済みの比較3種(共通10 ID・α ラダー `0 0.5 1 1.5`・色保持):
- `outputs/pretrained_ae/low_celeba_kc/` — CelebA 方向。個体を保ちつつ笑む。**推奨**。
- `outputs/pretrained_ae/low_celeba_feat_kc/` — 上 + `--feature-only`(平均顔への一律成分を除去)。低αでは A とほぼ同等。
- `outputs/pretrained_ae/low_fit_kc/` — FiT 方向。最も控えめ(物体の表情変化)。

**知見**: 低α(≲1.5)+色保持で「同じ顔」収束は消え、各物体が個性を保ったまま口元に笑みが出る。
強い笑顔がほしければ α を上げる/カスケードする(ただし人間顔に収束)。個体保存を最優先するなら低α。

## 依存

`diffusers` / `safetensors`(共有 `pyproject.toml` に追加済み)。SD-VAE 重み ~335MB を
HuggingFace から自動DL(初回のみ、ネットワーク必要)。
