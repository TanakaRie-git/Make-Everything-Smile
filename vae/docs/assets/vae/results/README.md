# 実験結果シート(overview)一覧

各実験の全10 ID コンタクトシート(共通の FacesInThings crop)。原本 PNG は
`vae/outputs/`(gitignore・再生成可能)。ここには表示用に縮小した JPG を追跡している。
詳細は各実験 README と [検証ログ](../../vae_verification_log.md)。

| ファイル | 実験 | 内容 | 列(α ラダー) |
|---|---|---|---|
| [scratch_baseline_overview.jpg](scratch_baseline_overview.jpg) | diffusion_compare (hq512) | スクラッチ VAE-GAN。object→人間笑顔へ収束 | 0 / 4 / 8 / 12 |
| [scratch_pareidolia_overview.jpg](scratch_pareidolia_overview.jpg) | diffusion_compare (pareidolia512) | 同上・+Pareidolia 学習済みモデル | 0 / 4 / 8 / 12 |
| [sdvae_celeba_overview.jpg](sdvae_celeba_overview.jpg) | pretrained_ae | 事前学習 SD-VAE + CelebA 方向。シャープだが人の笑顔が乗る | 0 / 4 / 8 / 12 |
| [sdvae_fit_kc_overview.jpg](sdvae_fit_kc_overview.jpg) | pretrained_ae | SD-VAE + FiT 方向 + 色保持。**物体自身が笑う** | 0 / 1 / 2 / 3 |
| [sdvae_cascade_overview.jpg](sdvae_cascade_overview.jpg) | pretrained_ae | 顔化→人間笑顔の2段カスケード。強い笑顔だが人間化 | face / 0 / 1 / 2 / 3 |
| [sdvae_low_celeba_kc_overview.jpg](sdvae_low_celeba_kc_overview.jpg) | pretrained_ae | CelebA 方向・低α・色保持。**個体を保ち控えめに笑む(推奨)** | 0 / 0.5 / 1 / 1.5 |
| [sdvae_low_celeba_feat_kc_overview.jpg](sdvae_low_celeba_feat_kc_overview.jpg) | pretrained_ae | 上 + feature-only | 0 / 0.5 / 1 / 1.5 |
| [sdvae_low_fit_kc_overview.jpg](sdvae_low_fit_kc_overview.jpg) | pretrained_ae | FiT 方向・低α・色保持。最も控えめ | 0 / 0.5 / 1 / 1.5 |

> 注: 学習済みチェックポイント(`*.pt`、各約108MB)は容量のため追跡しない。
> 全結果を再生成するには各実験 README のコマンドを実行する。
