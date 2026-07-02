# Make-Everything-Smile

顔ではない普通の物体に「笑った顔」を宿す（人工的な happy パレイドリアの生成）。
VAE / GAN / Diffusion の3手法で比較し、パレイドリア学習(Faces in Things)で自然さが
上がるかを検証するチーム課題。計画の全体像は
[vae/docs/make_everything_smile_plan.md](vae/docs/make_everything_smile_plan.md)。

## VAE トラック

本リポジトリは **VAE 担当分**。潜在空間の笑顔方向ベクトルで物体を編集する。
現行の本編は事前学習 SD-VAE を土台にした手法、変更前のスクラッチ VAE-GAN は
`vae/old/` にアーカイブしている。

→ **[vae/README.md](vae/README.md)**（手法・実験・構成）
