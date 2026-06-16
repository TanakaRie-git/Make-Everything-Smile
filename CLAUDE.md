# Make-Everything-Smile

「ただの物体への笑顔付与」を VAE / GAN / Diffusion の3手法で比較するプロジェクト。
全体計画は [docs/make_everything_smile_plan.md](docs/make_everything_smile_plan.md)。
このリポジトリでは VAE トラックを担当している([vae/](vae/)、[検証ログ](docs/vae_verification_log.md))。

## コミットメッセージ規約

- 1行目(サマリ)は **英語**。
- 本文(2行目以降)は **日本語** で内容を説明する。
- `Co-Authored-By` や `Generated with` のような共同作成者・生成元の表記は **入れない**。

例:
```
Add pixel-space blend mode to the smile editor

物体自身のシャープな画素を保ったまま、笑顔差分だけを目・口領域に重ねる。
スケール不一致・輪郭リーク・物体保存崩壊を同時に緩和する。
```

## 環境

- Python は **uv** で管理(`cd vae && uv sync`)。PyTorch は CUDA 12.4 ホイール。
- 実行環境は WSL2 (Ubuntu)、GPU は RTX 3070 Ti (8GB)。
- データ・成果物はリポジトリ直下の共有 `data/` `outputs/` に置く(gitignore 済み・再生成可能)。
