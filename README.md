# Make-Everything-Smile

顔ではない普通の物体に「笑った顔」を宿す(人工的な happy パレイドリアの生成)。
**VAE / GAN / Diffusion の3手法で同一入力・同一評価**の比較を行い、パレイドリア学習
(Faces in Things)で自然さが上がるかを検証するチーム課題。

![物体自身の顔が笑う(VAEトラックのキービジュアル)](vae/assets/current/cmp_object_smile_keyvis.jpg)
*左: 入力 → 右: 笑顔を強めた出力。物体の色・形を保ったまま、物体自身の口元が笑みになる(VAE トラック)。*

## トラック構成

| トラック | 場所 | 状態 | アプローチ |
|---|---|---|---|
| **VAE** | [vae/](vae/) → **[vae/README.md](vae/README.md)** | ✅ 実装済み | 凍結 SD-VAE の潜在空間に笑顔方向ベクトルを1本引き、係数 α で操作 |
| **GAN** | `gan/`(予定) |  | 表情を条件とした敵対的生成 |
| **Diffusion** | `diffusion/`(予定) |  | InstructPix2Pix + LoRA によるテキスト指示編集 |

評価は3手法共通の固定 10 crop([vae/experiments/eval_ids.txt](vae/experiments/eval_ids.txt))と
共通の軽量評価スクリプト([vae/experiments/eval.py](vae/experiments/eval.py)、LPIPS/SSIM/CLIP)で行う。

## VAE トラック

**追加学習ゼロ**: 事前学習 SD-VAE を凍結し、潜在に笑顔方向を足すだけ(`decode(encode(x) + α·σ·d)`)。
スクラッチ学習の VAE-GAN から土台を載せ替えることで、物体保存が LPIPS 0.74 → **0.17** に改善した。

| ① スクラッチ VAE-GAN | ② 同じ数式・凍結 SD-VAE に載せ替え | ③ FiT 方向 + 色保持 |
|---|---|---|
| ![scratch](vae/assets/current/cmp_scratch_humanize_bun.jpg) | ![sharp](vae/assets/current/cmp_pretrained_sharp_bun.jpg) | ![object-smile](vae/assets/current/cmp_object_smile_bun.jpg) |
| ボケ + 人間の顔に置換 | α=0 が完全にシャープに | 物体のまま自分の顔で笑う |

手法の詳細・クイックスタート・結果と知見は **[vae/README.md](vae/README.md)** へ。

## 環境

```bash
cd vae && uv sync   # PyTorch は CUDA 12.4 ホイール
```

WSL2 (Ubuntu) + RTX 3070 Ti (8GB) で動作確認。データ・成果物(`vae/data/` `vae/outputs/`)は
gitignore 済みで、各 README のコマンドから再生成できる。
