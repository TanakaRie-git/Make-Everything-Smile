# assets/ — 掲載画像の索引

`docs/` のテキストから参照する画像をここに集約する(原本 PNG は `vae/outputs/`・
gitignore・再生成可能。ここには表示用に縮小した JPG のみ追跡)。用途別に3分類:

## current/ — 現行手法(事前学習 SD-VAE)の作例

発表・README のキービジュアル。凍結 SD-VAE の潜在で笑顔方向を操作した結果。
ファイル名は `cmp_<段階>_<物体>` 系: `scratch_humanize`(スクラッチ版で人間化)/
`pretrained_sharp`(事前学習土台でシャープ化)/ `object_smile`(FiT方向+色保持で物体自身が笑う)。

**スライド「画質の弱点は土台の問題」用・同一物体(蒸しパン 000007868)で3段揃え**(推奨):

| ファイル | 内容 |
|---|---|
| [input_bun_000007868.jpg](current/input_bun_000007868.jpg) | 入力(素の蒸しパン) |
| [cmp_scratch_humanize_bun.jpg](current/cmp_scratch_humanize_bun.jpg) | スクラッチ版: ぼやけ + 人間化 |
| [cmp_pretrained_sharp_bun.jpg](current/cmp_pretrained_sharp_bun.jpg) | 事前学習土台 → 一気にシャープ |
| [cmp_object_smile_bun.jpg](current/cmp_object_smile_bun.jpg) | FiT方向+色保持で物体のまま笑う |

その他の作例(ワッフル 000015457 の3段 + 幕4のポテト/キービジュアル):

| ファイル | 内容 |
|---|---|
| [cmp_scratch_humanize.jpg](current/cmp_scratch_humanize.jpg) | ワッフル: 同一入力が人間の顔に化ける |
| [cmp_pretrained_sharp.jpg](current/cmp_pretrained_sharp.jpg) | ワッフル: 事前学習土台でシャープに |
| [cmp_object_smile_waffle.jpg](current/cmp_object_smile_waffle.jpg) | ワッフルが色・形のまま自分の口で笑う |
| [cmp_object_smile_potato.jpg](current/cmp_object_smile_potato.jpg) | ポテト、同上 |
| [cmp_object_smile_keyvis.jpg](current/cmp_object_smile_keyvis.jpg) | 発表キービジュアル(ワッフルの明確な笑み) |

候補差し替え用の別 ID セット(`input_` / `cmp_scratch_humanize_` / `cmp_pretrained_sharp_` /
`cmp_object_smile_` の各 `_000021539` / `_000021730` / `_000025426`)も current/ に同梱。

## old/ — 変更前スクラッチ VAE-GAN の作例

置き換え前の記録([../old/](../old/))。検証の時系列は
[../docs/vae_verification_log.md](../docs/vae_verification_log.md)。

| ファイル | 検証 | 内容 |
|---|---|---|
| [v2_faces_sanity.jpg](old/v2_faces_sanity.jpg) | v2 | 顔で笑顔方向が機能(サニティ) |
| [v2_backpack_collapse.jpg](old/v2_backpack_collapse.jpg) | v2 | グローバル潜在で物体が顔に崩壊 |
| [v3_faces_sanity.jpg](old/v3_faces_sanity.jpg) | v3 | 空間潜在+男女別方向 |
| [v3_icecream.jpg](old/v3_icecream.jpg) | v3 | α による強度遷移 |
| [hq512_golf_scale_contour.jpg](old/hq512_golf_scale_contour.jpg) | hq512 | スケール不一致・輪郭リーク |
| [hq512_fit_recon_ood.jpg](old/hq512_fit_recon_ood.jpg) | pareidolia512 | FiT の再構成崩れ |
| [blend_gas_pump.jpg](old/blend_gas_pump.jpg) | blend | ピクセルブレンドで物体保存 |
| [fruit_lemons.jpg](old/fruit_lemons.jpg) | fruit | 単一果物に笑顔が宿る(本命作例) |
| [fruit_apples.jpg](old/fruit_apples.jpg) | fruit | 同上(リンゴ) |
| [fruit_avocados_fit.jpg](old/fruit_avocados_fit.jpg) | fruit | 同上(アボカド・FiT 方向) |

## results/ — 全 ID overview シート + 評価メトリクス

共通10 ID(FacesInThings crop)のコンタクトシートと軽量評価の CSV。
各シートの実験・列(α ラダー)の対応表は [results/README.md](results/README.md)。
