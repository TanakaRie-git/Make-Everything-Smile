# 引き継ぎ書: Smile Diffusion 実装

## プロジェクト概要

MM-CelebA-HQ データセットを使い、**非笑顔の顔画像を笑顔に変換する** モデルを
InstructPix2Pix (IP2P) + LoRA fine-tuning で実装する。

- リポジトリ: `~/work/Make-Everything-Smile/`
- 担当ディレクトリ: `~/work/Make-Everything-Smile/diffusion/`
- 他に `vae/`, `gan/` ディレクトリがあり、別担当者が別アプローチで同じ課題に取り組んでいる
  （**diffusion ディレクトリ以外は触らないこと**）
- パッケージ管理: `uv`（**ワークスペースではなく、diffusion 単体で独立した
  `pyproject.toml` / `uv.lock` / `.venv` を持つ**。torch のバージョンが
  vae/gan チームと異なるため、意図的に分離している）

## 実行環境

- ローカル RTX 5080（研究室サーバー、CUDA 12.8 系）
- OS: WSL2 (Ubuntu) on Windows
- Python 3.10+

## 既存資産

**何もない。`diffusion/` ディレクトリの中身はゼロから実装する。**
以下は「こういう構成・設計で作ってほしい」という仕様であり、
完成コードではない。

```
diffusion/
├── pyproject.toml          # 依存関係定義（torch, diffusers, peft 等）
├── configs/
│   └── train_lora.yaml     # 学習設定
├── data/
│   ├── build_pairs.py      # ペアデータ構築スクリプト
│   └── dataset.py          # PyTorch Dataset
├── scripts/
│   ├── train.py            # 学習メインスクリプト
│   └── infer.py            # 推論スクリプト
└── README.md
```

このタスクの目的は、上記を新規実装し、実際にデータを通して
学習が走ることを確認すること。

## やってほしいこと（タスク一覧）

### 1. 環境セットアップの確認

```bash
cd ~/work/Make-Everything-Smile/diffusion
uv sync
```

- CUDA 12.8 用 torch が正しく入るか確認する
- 直前に WSL2 のディスク容量問題で `uv sync` が失敗していた経緯があるため、
  まずディスク容量に余裕があることを前提として進めてよい
  （解消済みとして引き継ぐ）

### 2. データ準備

CelebA-HQ 画像・属性ファイルは以下を想定（パスは実際の配置場所に合わせて読み替えること）：

```
/path/to/celeba_hq/
├── CelebA-HQ-img/
├── CelebAMask-HQ-attribute_list.txt
└── CelebA-HQ-to-CelebA-mapping.txt   # あれば使う
```

```bash
uv run python data/build_pairs.py \
  --data_root /path/to/celeba_hq \
  --max_pairs 10000 \
  --val_ratio 0.1
```

`data/build_pairs.py` の実装方針と既知の制約：

- **重要な設計判断**: CelebA-HQ には同一人物の笑顔/非笑顔ペアが存在しない。
  そのため `smiling=0` の人と `smiling=1` の人を**別々にランダムサンプリング
  してペアにする**実装にすること（同一人物ペアは作れない前提）。
  - CelebA-HQ の属性ファイル（`Smiling` 列）を使って分類する
  - これでまず動作確認を進めて問題ない
  - もし結果の品質が低い場合、`identity_CelebA.txt` を使って
    同一人物内でのペアリングに変更する拡張を検討する
    （今は実装不要、将来の拡張として TODO コメントを残しておく）

### 3. 学習スクリプトの動作確認とデバッグ

```bash
uv run python scripts/train.py --config configs/train_lora.yaml
```

`scripts/train.py` の実装方針と重要な注意点：

- **IP2P の U-Net 入力仕様に注意**: 通常の Stable Diffusion と異なり、
  U-Net への入力は `[noisy_target_latent (4ch), original_image_latent (4ch)]`
  を channel 方向に concat した **8ch** にする必要がある。
  - base model は `timbrooks/instruct-pix2pix`（diffusers経由）を使うこと。
    通常の SD 1.5 などを誤ってベースにすると in_channels=4 のままになり
    shape mismatch が起きるので注意。

- **解像度とバッチサイズ**: まず `resolution: 256`, `train_batch_size: 4`
  で動作確認すること。512px はメモリ要件が大きく上がるため、
  256px で学習・推論が一通り通ってから初めて上げる。

- **LoRA 設定**: U-Net の attention 層 (`to_k`, `to_q`, `to_v`, `to_out.0`)
  のみに peft 経由で LoRA を適用する設計にすること。VAE / text encoder は凍結する。

- `configs/train_lora.yaml` 内の `data_root` / `train_pairs` / `val_pairs`
  は実際のデータ配置パスに合わせて設定すること（後述）。

### 4. 推論スクリプトの動作確認

```bash
uv run python scripts/infer.py \
  --lora_path outputs/smile_lora_v1/final/unet_lora \
  --input_image <テスト用の顔画像パス> \
  --output_dir outputs/results
```

- `image_guidance_scale`（元画像保持の強さ）と `guidance_scale`
  （テキスト条件の強さ）はデフォルト 1.5 / 7.5 から始めて、
  identity が崩れる場合は `image_guidance_scale` を 2.0〜2.5 に上げて調整する。

### 5. 動作確認後にやってほしいこと

- 学習が一通り通ったら、`outputs/results/` に入出力比較画像が
  生成されることを確認する
- 256px で動いたら 512px に解像度を上げて再学習し、VRAM 使用量と
  学習時間を記録する
- 学習ログ（TensorBoard）が `outputs/smile_lora_v1/logs/` に
  正しく出力されているか確認する

## 完了の定義（Done の条件）

1. 上記構成のファイル一式が `diffusion/` 以下に実装されている
2. `uv sync` が通り、`.venv` が作成される
3. `build_pairs.py` が `pairs/train.json`, `pairs/val.json` を生成する
4. `train.py` がエラーなく最低1エポック走り、チェックポイントが
   `outputs/` 以下に保存される
5. `infer.py` で任意の顔画像を入力し、笑顔に変換された画像が
   出力される（品質の良し悪しは問わない。パイプラインが通ることが目標）
6. 256px での動作確認が完了している（512px は時間があれば）

## 触ってはいけないもの

- `vae/`, `gan/` ディレクトリ（別担当者の作業領域）
- リポジトリルートの `pyproject.toml`（もし存在する場合。
  diffusion は独立した uv 環境のため、ルートに依存定義を書く必要はない）

## 不明点があった場合

実装上の判断に迷った場合（特に identity ペアリングの扱いや、
LPIPS 損失を追加するかどうかなど）は、仮の判断をして進めつつ、
コメントで TODO として明示すること。