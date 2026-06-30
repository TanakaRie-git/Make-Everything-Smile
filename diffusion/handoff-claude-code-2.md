# 引き継ぎ書: Concept Slider (DCGAN風ベクトル演算をDiffusionで実装)

## 背景・目的

現在 `diffusion/` で IP2P + LoRA による笑顔変換を実装しているが、
学習データが「ランダムな非笑顔の人」と「ランダムな笑顔の人」のペアに
なっているため、**Aさんの顔を入力するとBさん（別人）の笑顔が出力される**
というidentity崩壊の問題が起きている。

DCGAN論文 (Radford et al., 2016, arXiv:1511.06434) が示した
「潜在空間でのベクトル演算」（`smile_vector = mean(z_smiling) - mean(z_neutral)`,
`z_A + alpha * smile_vector` でAさんの笑顔を生成）と同じ発想を、
diffusionモデル上で **Concept Slider** という手法を使って実装する。
これは画像ペアを使わず、プロンプトの対比だけで「笑顔方向」を学習する
ため、既存のIP2Pパイプラインのidentity崩壊問題を構造的に回避できる。

## 配置場所

- リポジトリ: `~/work/Make-Everything-Smile/`
- **新規ディレクトリ `concept_slider/` を作成して実装する**
  （`diffusion/`, `vae/`, `gan/` とは独立した別ディレクトリ。
  既存の `diffusion/` の中身は変更しない）
- パッケージ管理: `uv`。`concept_slider/` 配下に独立した
  `pyproject.toml` / `uv.lock` / `.venv` を持たせる
  （他ディレクトリとtorchバージョンが揃わなくても問題ない構成）

## 実行環境

- ローカル RTX 5080（研究室サーバー、CUDA 12.8系）
- OS: WSL2 (Ubuntu) on Windows
- Python 3.10+

## DCGANとの対応関係（実装の核となる考え方）

| DCGAN | Concept Slider (diffusion) |
|---|---|
| `z`（潜在ベクトル） | `z_T`（DDIM Inversionで得る初期ノイズ） |
| `mean(z_smiling) - mean(z_neutral)` | LoRAが学習する方向（プロンプト対比のノイズ予測差分） |
| `z_A + alpha * smile_vector` | `z_T(A)` を起点にLoRAスケール`alpha`でdenoise |
| Generator(編集後のz) | U-Net + VAE decoder |

DCGANは画像→潜在ベクトルの逆変換が前提だったが、これはdiffusionでは
**DDIM Inversion** が担う。decisiveな違いは、DCGANのzが1回のforward
passで決まるのに対し、diffusionは50ステップ程度のdenoise過程全体に
方向ベクトルを適用する点。

## ディレクトリ構成（新規実装する）

```
concept_slider/
├── pyproject.toml
├── configs/
│   └── slider.yaml          # 学習設定（プロンプト対比リスト含む）
├── scripts/
│   ├── train_slider.py      # LoRAで方向ベクトルを学習
│   └── infer_slider.py      # DDIM Inversion + slider適用で推論
└── README.md
```

## やってほしいこと（タスク一覧）

### 1. 環境セットアップ

```bash
cd ~/work/Make-Everything-Smile/concept_slider
uv sync
```

CUDA 12.8用のtorchが正しく入るか確認する。

### 2. `train_slider.py` の実装方針

画像データは不要。学習の核心は以下のロジック：

```python
# LoRAをOFF（ベースモデル）にした状態で
# target（笑顔）プロンプトとneutral（無表情）プロンプトの
# ノイズ予測の差分を計算する。これがDCGANのsmile_vectorに相当する。
direction = (eps_target - eps_neutral).detach()

# anchorプロンプト（一般的な人物プロンプト）に対し、
# LoRAをONにした予測がこの方向にscale倍だけ移動するように学習する
target_shifted = eps_anchor_base + scale * direction
loss = F.mse_loss(eps_anchor_with_lora, target_shifted)
```

実装上の重要なポイント：

- **画像は使わずランダムノイズ（`torch.randn`）を起点に学習する**。
  プロンプトの対比だけで方向を学習するのがConcept Sliderの核心。
- LoRAは `peft` の `LoraConfig` でU-Netのattention層
  (`to_k`, `to_q`, `to_v`, `to_out.0`) にのみ適用する。
- rankは低くてよい（4程度）。方向ベクトル1本を学習するだけなので
  通常のfine-tuningよりパラメータ数は少なくて済む。
- `unet.disable_adapter()` コンテキストでLoRA OFF時の予測を取得し、
  LoRA ON時の予測と比較する設計にすること。
- 学習設定（`configs/slider.yaml`）にはプロンプトの対比リストを
  複数バリエーション用意する（"a photo of a person smiling",
  "a photo of a person with a big smile" など）。バリエーションが
  少ないと方向ベクトルの汎化性能が落ちる。

学習データの構成イメージ：

```yaml
prompts:
  target:   # 笑顔方向
    - "a photo of a person smiling"
    - "a photo of a person with a big smile"
    - "a photo of a happy smiling face"
  neutral:  # 無表情方向
    - "a photo of a person with a neutral expression"
    - "a photo of a serious face"
  anchor:   # identity保持の基準点
    - "a photo of a person"
    - "a portrait photo of a face"
```

### 3. `infer_slider.py` の実装方針

推論は2段階。

**Step 1: DDIM Inversion**（Aさんの画像を潜在表現 `z_T` に逆変換）

```python
# LoRAをOFFにした状態でDDIM Inverse Schedulerを使い、
# 画像を決定論的に初期ノイズ z_T へ逆変換する
# （DCGANの「画像→潜在ベクトルz」の逆変換に相当）
```

重要：inversion自体は **LoRAをOFFにした状態（ベースモデルの表現空間）**
で行うこと。LoRAをONにしたままinversionすると方向ベクトルが
混入して逆変換の精度が落ちる。

**Step 2: Slider適用のdenoise**

```python
# z_T を起点に、LoRAのスケール（adapter weight）を
# alpha として設定してdenoiseする
# scale=0なら元画像に戻る、scale>0で笑顔が強くなる
pipe.unet.set_adapters(["default"], weights=[scale])
```

- `--scale` パラメータがDCGANの `alpha` に相当する。0、1、2、3、4の
  ように複数のscaleを並べた比較画像を自動生成する機能を入れること
  （`outputs/slider_scale_comparison.png` のような形式）。
- `num_inference_steps` はデフォルト50。精度を上げたい場合は100に
  増やせるが推論時間も伸びる。

### 4. 動作確認

```bash
# まずステップ数を減らして動作確認（例: max_train_steps=100）
uv run python scripts/train_slider.py --config configs/slider.yaml

# 推論
uv run python scripts/infer_slider.py \
  --slider_path outputs/smile_slider_v1/final \
  --input_image <テスト用の顔画像> \
  --scale 2.0
```

### 5. 既知の論点・注意が必要な箇所

- **LoRAの保存・読み込み形式**: `peft` で保存したLoRAを
  `pipe.unet.load_attn_procs()` で読み込もうとすると形式が合わない
  可能性がある。読み込みエラーが出た場合は `PeftModel.from_pretrained`
  系のAPIに変更すること。
- **DDIM Inversionの精度**: 完璧な逆変換ではないため、複雑な背景や
  特殊なポーズでは元画像からわずかにズレることがある（GAN inversion
  と同様の制約として許容する）。
- **scaleの効き方**: anchorプロンプトと実際の入力画像の分布が
  ズレている場合、scaleの値域（適切な笑顔の強さが出る範囲）が
  事前の想定と異なる可能性がある。0〜5程度の範囲で実際に試して
  調整すること。

## 完了の定義（Done の条件）

1. `concept_slider/` 以下に上記構成のファイル一式が実装されている
2. `uv sync` が通り、`.venv` が作成される
3. `train_slider.py` がエラーなく走り、`outputs/smile_slider_v1/final`
   にLoRAの重みが保存される
4. `infer_slider.py` で任意の顔画像を入力し、DDIM Inversion → slider適用
   → 笑顔画像が出力される
5. scale=0で出力が元画像とほぼ同一であること（identityが保持される
   ことの確認）、scaleを上げると同一人物のまま笑顔が強くなることを
   確認する
6. 既存の `diffusion/` のIP2P出力（別人になってしまう問題）と比較して、
   identity保持の改善が見られるか確認する

## 触ってはいけないもの

- `diffusion/`, `vae/`, `gan/` ディレクトリ（既存の実装・別担当者の領域）
- リポジトリルートの `pyproject.toml`（存在する場合）

## 不明点があった場合

LoRAの保存形式、DDIM Inversionの実装詳細（`diffusers` のAPIバージョン
差異）など、判断に迷う箇所は仮実装を進めつつTODOコメントを残すこと。
特に「画像ペアを使った拡張」（プロンプトだけでなく実際の笑顔/非笑顔の
画像から方向を学習する発展形）は今回のスコープ外なので、必要になれば
別タスクとして切り出す。