# 実験: Diffusion トラックとの同一条件比較

Diffusion トラック(`diffusion/`)は FacesInThings のパレイドリア crop に対し、
IP2P+LoRA / Concept Slider で笑顔を付与している。本実験は **同じ crop・同じ ID・
同じ出力プロトコル(scale ラダー)** で VAE-GAN の潜在編集を回し、手法間で
「どう笑顔が宿るか / 物体を保つか・人間化するか」を並べて比較する。

VAE の編集は `z = mu(E(x)); D(z + alpha * proj_std * direction)`。これは DCGAN 由来の
潜在ベクトル演算で、Diffusion の Concept Slider の scale ラダーと同型。alpha を上げると
**物体のまま → 目・口がうっすら宿る → 顔が支配的(人間化)** と連続遷移する
(スモークテストの知見。[../objects/README.md](../objects/README.md))。

## 共通入力(固定評価セット)

- [eval_ids.txt](eval_ids.txt) — Diffusion の `outputs/` が compare を出した ID と、
  FacesInThings crop キャッシュの積集合。全手法で同一 ID を使う(計画 §0.3)。
- crop 本体は `vae/data/diffusion_compare/inputs/`(gitignore 済み)にコピー済み。
  Diffusion 側 `diffusion/data/cache/facesinthings_crops/` と同一ファイル。

## 実行(cd vae から / 2条件)

学習済みチェックポイントは本編実験のものを再利用する(追加学習不要)。

```bash
# Baseline(CelebA-HQ + 物体で学習)
uv run python experiments/diffusion_compare/compare_facesinthings.py \
  --ckpt outputs/hq512/ckpt/last.pt \
  --direction outputs/hq512/smile_direction.pt \
  --input-dir data/diffusion_compare/inputs \
  --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/diffusion_compare/baseline \
  --scales 0 4 8 12

# +Pareidolia(上記 + FacesInThings happy で fine-tune)
# 方向は Baseline と揃える(§0.2 は学習データだけ変える)。pareidolia512 の
# CelebA 笑顔方向は下記で一度だけ算出しておく:
#   uv run python -m smilevae.direction \
#     --ckpt outputs/pareidolia512/ckpt/last.pt \
#     --pair data/raw/celeba_hq/smile_male   data/raw/celeba_hq/neutral_male \
#     --pair data/raw/celeba_hq/smile_female data/raw/celeba_hq/neutral_female \
#     --out outputs/pareidolia512/smile_direction.pt
uv run python experiments/diffusion_compare/compare_facesinthings.py \
  --ckpt outputs/pareidolia512/ckpt/last.pt \
  --direction outputs/pareidolia512/smile_direction.pt \
  --input-dir data/diffusion_compare/inputs \
  --ids experiments/diffusion_compare/eval_ids.txt \
  --out-dir outputs/diffusion_compare/pareidolia \
  --scales 0 4 8 12
```

## 出力(diffusion の作法に対応)

| ファイル | 中身 | diffusion 側の対応 |
|---|---|---|
| `<id>_compare.png` | `input \| α=0 \| α=… ` の横並び | `<id>_compare.png` |
| `<id>_smile.png` | 最大 α の笑顔画像 | `<id>_smile.png` |
| `_overview.png` | 全 ID のコンタクトシート | — |

## 方向ベクトルの選択(効き方の違い)

- `smile_direction.pt` — CelebA の笑顔方向。**人間の笑顔に振れやすい(humanize しやすい)**。
  §0.2 の 2条件比較はこちらで揃える(方向を固定し学習データだけ変える)。
- `fit_direction.pt` — FacesInThings happy 方向。物体っぽさを保つ狙いだが、
  **pareidolia512 の decoder では α≥4 で格子状ノイズに崩壊**する(proj_std が大きく
  過剰に押すため)。使う場合は `--scales 0 1 2 3` 程度に絞ること。

## α ラダーの目安

パレイドリア crop は顔多様体からやや離れた OOD 入力なので、顔用の α=1〜3 より強めが要る。
`smile_direction` では `0 4 8 12` がちょうど良い(alpha=0 は recon=identity 確認)。

## 観察された挙動(初回ラン、共通10 ID)

- **両条件とも α を上げると object → 人間の笑顔へ連続遷移**する。パレイドリア crop が
  そのまま人間顔になるのは想定内の結果(高 α 側)で、失敗ではない。
- **Baseline(hq512)** は年配男性寄りの顔に、**+Pareidolia(pareidolia512)** は別の
  笑顔顔になり、笑い方(歯の見え方)も変わる。物体保存は両者とも弱く、crop が顔枠で
  切られているため α=4 で既に顔が支配的。
- VAE 特有の**ボケ**が全域で出る(diffusion のシャープさとの対比が比較の主眼)。
- 出力原本: `outputs/diffusion_compare/{baseline,pareidolia}/`(gitignore 済み)。
