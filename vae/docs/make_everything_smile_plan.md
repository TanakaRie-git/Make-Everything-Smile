# プロジェクト計画書
## ただの物体への「笑顔」付与 — VAE/GAN/Diffusion 比較 ＋ パレイドリア学習による自然さ向上の検証

> **📌 実装反映(2026-07-02 更新):** 本書は当初計画。実装の進行で変わった点は本文中に
> **[実装]** として注記した。VAE トラックで確定した主な差分(経緯は [検証ログ](vae_verification_log.md)):
> - **VAE の手法**: スクラッチ学習の VAE-GAN は再構成のボケが天井となり [old/](../old/) にアーカイブ。
>   現行は **事前学習 SD-VAE(`stabilityai/sd-vae-ft-mse`)を凍結し、その潜在で笑顔方向を操作**
>   ([experiments/pretrained_ae/](../experiments/pretrained_ae/))。
> - **入力・評価ドメイン**: ImageNet 非生物のフルキュレーションは未実施。評価入力は
>   **Faces in Things のパレイドリア crop(Diffusion トラックと同一)** に変更。ImageNet 系は
>   代替セット(Tiny ImageNet / Imagenette / Fruits-30)をスクラッチ期の学習にのみ使用。
> - **解像度**: 256×256 → **512×512**(Diffusion トラックと統一)。
> - **評価セット**: 固定100枚 → まず **共通固定 10 ID**([experiments/eval_ids.txt](../experiments/eval_ids.txt))
>   で軽量評価。100枚化と FID/KID(数百枚必要)は未実施。
> - **笑顔強度**: 弱・中・強の3段階 → **α ラダー**(例 `0 1 2 3`、α=0 は再構成)。
> - **+Pareidolia 介入**: スクラッチ期は計画どおり学習データ追加(hq512 vs pareidolia512)で実施。
>   現行 SD-VAE は decoder 凍結で追加学習をしないため、介入は **笑顔方向の学習ソース**
>   (CelebA 人間笑顔 `smile_direction` vs FiT happy−neutral `fit_direction`)として実現。
> - **評価指標**: LPIPS/SSIM/CLIP の軽量評価を実装([experiments/eval.py](../experiments/eval.py))。
>   保存軸は機能する一方 **CLIP-smile はほぼノイズで「人間化」を報酬化** → 笑顔軸は人手評価が
>   必須(§0.1-5 の想定どおり)。

---

## 0. プロジェクトの目的と前提整理

**ゴール:** 顔ではない普通の物体に、「笑っている顔」が自然に宿って見えるように加工する（＝人工的に happy なパレイドリアを作る）。VAE・GAN・Diffusion の3手法で生成し、「どの手法が最も自然に笑顔を付与できるか」を比較する。さらに、**パレイドリア画像（Faces in Things）を学習に加えると自然さが向上するか**を検証する。最終成果物は**発表スライド（PowerPoint）**。論文執筆はしない。

**チーム:** 3名（各自が1手法ファミリーを担当し、各自が2条件を回す）。

### 0.1 着手前に合意すべき技術的論点
1. **「笑顔付与」の意味＝物体が笑った顔に見えるように加工する。** 物体の形は保ったまま、目＋笑った口が自然に宿るようにする。
2. **物体にも顔のランドマークは無い。** 笑顔をどこに付けるかは別途決める（→ 2.4、まずは全体編集）。
3. **before/after のペアデータが無い。** 教師あり変換（pix2pix系）は不可。unpaired／属性編集で扱う。
4. **「笑顔」概念は実顔から学ぶしかない。** 物体には笑顔ラベルが無いので、CelebA-HQ の `Smiling` を全手法共通の供給源にする。
5. **「自然さ」が本PJの主役。** パレイドリア介入の有無で自然さがどう動くかが中心結果。評価は人手を金標準にする（実顔用分類器は人工パレイドリアで発火しにくい）。

### 0.2 中心的な実験設計（パレイドリア介入のアブレーション）
各手法について **2条件**を必ず実施：

| 条件 | 学習データ | 狙い |
|---|---|---|
| **Baseline** | CelebA（笑顔）＋ 物体データ。パレイドリアなし | 素の笑顔付与の自然さ |
| **+Pareidolia** | 上記 ＋ Faces in Things | 自然さが増すか |

**ヘッドライン質問：** ①どの手法が一番自然に笑顔を付けられるか／②パレイドリア学習で自然さは上がるか・どの手法が一番恩恵を受けるか。

### 0.3 公平な比較のための共通設定
| 項目 | 設定 |
|---|---|
| 笑顔の教師 | CelebA-HQ `Smiling`（事前学習済み重みOK、スクラッチ学習はしない） |
| 入力・評価ドメイン | 物体データ（2.1で確定）→ **[実装]** Faces in Things のパレイドリア crop（Diffusion と同一） |
| 自然さ参照分布 | Faces in Things の `happy` サブセット |
| 解像度 | 256×256 で統一 → **[実装]** 512×512（Diffusion と統一） |
| 笑顔強度 | 弱・中・強の3段階 → **[実装]** α ラダー（例 `0 1 2 3`、α=0=再構成） |
| 評価セット | 物体画像の共通100枚（全手法・全条件で同一）→ **[実装]** まず共通固定 10 ID（[eval_ids.txt](../experiments/eval_ids.txt)） |
| 乱数シード | 評価サンプルは固定 |

---

## 1. フェーズ1 — 関連研究の列挙
「課題設定 / 手法 / 評価指標 / 本PJへの示唆」の4点でメモ化する。

### 1.1 パレイドリア（自然さの参照・介入の根拠）
- Hamilton et al., *Seeing Faces in Things*, arXiv:2409.16143 (ECCV 2024) — happy な“物体顔”の実例集。自然さの参照＆学習介入に使う。
- パレイドリック画像生成の先行例（自然さ不足が課題）— 本PJの新規性の置き所。

### 1.2 表情編集・笑顔生成 GAN
- StarGAN / StarGAN v2（unpaired ドメイン変換）、GANimation（AU連続制御、笑顔強度）、AttGAN/STGAN（属性編集）、CycleGAN。
- StyleGAN系の潜在編集（InterFaceGAN：笑顔方向ベクトル）。

### 1.3 VAE・潜在編集
- CVAE、β-VAE/FactorVAE（解きほぐし）、VAE-GAN（ぼやけ補正＝自然さに重要）、潜在空間の属性方向操作。

### 1.4 Diffusion ベースの編集
- DiffAE（意味的潜在で潜在編集、VAE/GANと同枠比較）、SDEdit（ノイズ強度で編集強度制御）、InstructPix2Pix（指示編集の強い既製ベースライン）、ControlNet/Prompt-to-Prompt/Imagic（局所・構造制約）。

### 1.5 評価指標・知覚研究
- FID/KID、LPIPS/SSIM/PSNR、AU検出（OpenFace AU6/AU12）、笑顔分類器、主観評価（MOS/2AFC）。

**成果物:** 関連研究マトリクスと本PJのポジショニング（「ただの物体への自然な笑顔付与 × パレイドリア学習介入」という空きスロット）。

---

## 2. フェーズ2 — データセットの決定

### 2.1 採用データセット（3つの役割）
| 役割 | データセット | 用途 |
|---|---|---|
| ★入力・評価先 | **ImageNet（生物以外）【当初計画】** WordNet階層で animal/person/plant 配下を除外し、50〜100クラス×各100〜200枚（1〜3万枚、5〜15GB）をキュレーション。全量138GBは不要 | 笑顔を付与し、自然さを評価。学習にも使う |
| 笑顔の教師 | **CelebA-HQ**（`Smiling`） | 笑顔概念の供給源 |
| ★自然さ介入＆参照 | **Faces in Things**（arXiv 2409.16143） | +Pareidolia条件の追加学習データ／happyサブセットをFID基準に |

> クラス選定は Faces in Things に出る物の種類（家庭用品・食品・建物・乗り物など）に寄せると比較がきれいになる。

**[実装]** ImageNet 本体のフルキュレーションは行わなかった。実際に使ったのは:
- **学習用の物体（スクラッチ VAE-GAN 期のみ）**: 代替セットとして Tiny ImageNet 非生物
  （WordNet 除外ロジックは計画どおり実装、[scripts/download_objects_smoke.py](../scripts/download_objects_smoke.py)）、
  Imagenette 非生物8クラス（512px 用、[scripts/download_objects_hq.py](../scripts/download_objects_hq.py)）、
  Fruits-30 単一物体（作例用、[scripts/download_objects_single.py](../scripts/download_objects_single.py)）。
- **評価入力**: ImageNet 系ではなく **Faces in Things のパレイドリア crop**（Diffusion トラックと
  同一 crop・同一 10 ID）に変更。現行 SD-VAE 手法は追加学習をしないため、物体学習データ自体が不要。

### 2.2 前処理・分割
- 物体画像：物体中心にクロップ → 256×256 正規化（アスペクト比保持＋白パディングでFaces in Thingsと作法を揃える）。**[実装]** 白パディング正方形化は実装どおり（`src/smilevae/data.py`）、解像度は 512×512。
- 評価用の固定100枚を切り出し、全手法・全条件で共通利用。**[実装]** まず共通固定 10 ID（[eval_ids.txt](../experiments/eval_ids.txt)）で運用、100枚化は未実施。
- Faces in Things：emotion属性で `happy` を抽出（自然さ参照＆+Pareidolia学習に）。**[実装]** 実装済み（[scripts/download_faces_in_things.py](../scripts/download_faces_in_things.py)、happy 1216枚）。

### 2.3 Baseline / +Pareidolia の作り分け
- Baseline学習集合 ＝ CelebA(笑顔) ＋ 物体データ。
- +Pareidolia学習集合 ＝ Baseline ＋ Faces in Things（happy中心）。
- 2条件で**学習データ以外（解像度・反復数・評価セット）は揃える**。差が介入のみに帰着するように。

**[実装]** スクラッチ VAE-GAN 期は計画どおり学習データ追加で2条件を実施
（Baseline=hq512 / +Pareidolia=pareidolia512、[検証ログ §4–5](vae_verification_log.md)）。
現行の SD-VAE 手法は decoder 凍結で追加学習をしないため、介入は
**笑顔方向ベクトルの学習ソース**の差し替えとして実現:
Baseline 相当 = CelebA 人間笑顔から算出した `smile_direction`、
+Pareidolia 相当 = FiT の happy−neutral から算出した `fit_direction`
（編集パイプライン・評価セットは両者で同一）。

### 2.4 「笑顔をどこに付けるか」
- 方針A（主軸・全員）：物体全体を happy パレイドリア顔へ寄せる（領域指定なし）。
- 方針B（余力時）：顔を宿す位置を指定して局所生成。

**成果物:** データ仕様書（物体データ確定、前処理スクリプト、固定100枚リスト、2条件の学習集合定義）。

---

## 3. フェーズ3 — 各手法による生成

### 3.1 共通パイプライン
```
[CelebA: Smiling] ─学習─▶「笑顔」概念
[物体データ] ────────────▶ 笑顔付与の対象＆物体の見た目
       （+Pareidolia条件のみ Faces in Things を学習に追加）
            │
            ▼
[ただの物体(入力)] ─適用─▶ [笑った顔が宿った物体(出力)]
```
入出力・解像度・評価セットは 0.3 に従う。各自 Baseline と +Pareidolia の両方を出力。

### 3.2 手法別設計（担当 = 各1名）
- **VAE：** CVAE / VAE-GAN（ぼやけ対策にGAN損失推奨）。笑顔方向ベクトルを潜在で操作。
  **[実装]** VAE-GAN（Larsen et al. 2016）で笑顔方向操作まで実証したが、スクラッチ学習の
  decoder が再構成ボケの天井となり [old/](../old/) にアーカイブ。現行は
  **事前学習 SD-VAE（凍結）の潜在で同じ方向操作**を行う
  （[experiments/pretrained_ae/](../experiments/pretrained_ae/)）。「笑顔方向ベクトルを潜在で操作」
  という設計自体は計画どおり。
- **GAN：** StarGAN v2（neutral↔happyドメイン変換）本命、GANimation（AU強度）を比較軸に。
- **Diffusion：** DiffAE（潜在編集）＋ SDEdit / InstructPix2Pix（既製ベースライン）。
  **[実装]** InstructPix2Pix ベースに **FiT の neutral→happy ペアで LoRA を学習**する方式
  （SD-VAE は凍結。`diffusion/configs/train_lora.yaml`, `diffusion/data/build_pairs_pareidria.py`）。

### 3.3 アブレーション（全手法共通）
- 笑顔強度（弱・中・強）。
- **Baseline vs +Pareidolia（本PJの主役）。**
- 方針A vs B（余力時）。

**成果物:** 各手法×2条件×強度の生成画像、学習設定ログ（計算予算含む）。

---

## 4. フェーズ4 — 評価（「自然さ」が主役）

### 4.1 評価軸
| 軸 | 問い | 指標 |
|---|---|---|
| ① 笑顔度 | 笑った顔が付いたか | happy/neutral分類器の確率（Faces in Things感情ラベルで較正）、AU6/AU12（補助） |
| ②★ 自然さ | 自然な物体顔か | FID/KID（Faces in Things happy基準） |
| ③ 物体保存 | 元の物体と分かるか | 編集前後 LPIPS/SSIM |
| ④ 顔に見えるか | 顔として読めるか | 原論文パレイドリア検出器の信頼度 |

**[実装]** 軽量評価 [experiments/eval.py](../experiments/eval.py)（共通10 ID）まで実装済み:
- **③物体保存 = LPIPS/SSIM/CLIP画像類似**は計画どおり実装し、綺麗に機能（変種の順位が明確に出る）。
- **①笑顔度**は分類器/AU の代わりに **CLIP-smile** を試行したが、**ほぼノイズ（±0.01）で
  「人間化する手法」だけを報酬化**し、物体を保った笑顔を過小評価 → 笑顔軸は人手評価（§4.3）が必須。
- **②FID/KID・④パレイドリア検出器は未実施**（FID/KID は数百枚必要、現状10枚のため）。

### 4.2 介入効果の見せ方
- **Baseline vs +Pareidolia の②自然さ（FID＋人手）の差分**がヘッドライン。手法別に「介入で自然さがどれだけ動いたか」を棒グラフ＋作例で。
- ①笑顔度↔②自然さ↔③物体保存 はトレードオフ。強度を振った曲線でPareto的に提示。

### 4.3 主観評価（金標準）
- 2AFC：自然さ／笑顔らしさを別設問。さらに **同一入力で Baseline vs +Pareidolia のどちらが自然か**の選好を直接取る。
- 評価者10名以上、固定100枚。Bradley–Terry等でランキング。
- 自動指標②と人手評価の順位相関（Spearman）を報告。

**成果物（Phase 4）:** 発表スライド（PowerPoint）。①手法×条件の比較表、②自然さの介入効果（差分）、③ビフォー/アフター作例、④人手評価ランキング、⑤自動指標と人手の相関と結論。※論文執筆はスコープ外。

---

## 5. スケジュール（3人 × 4週間スプリント）

**ゴール：実際に動かし、「どの手法がどれだけ自然に笑顔を付けられたか」「パレイドリア学習で自然さが上がったか」を発表スライドにまとめるまで。**

### 5.0 4週間で間に合わせる鉄則
- **スクラッチ学習をしない。** CelebA-HQ事前学習済みを起点に、適用・微調整だけ。（**[実装]** VAE トラックは当初この鉄則に反してスクラッチ学習し、ボケの天井に当たった。事前学習 SD-VAE への載せ替えで解消——鉄則の正しさを裏づける結果に。）
- **評価は物体100枚に固定。**（**[実装]** まず共通固定 10 ID で軽量評価から。）
- **方針A（全体編集）を主軸**、方針Bは余力時。
- **2条件は学習データ違いのみ。** パイプラインを使い回し、+Pareidolia は追加データを足すだけにする。
- 物体データのキュレーション・共通前処理・評価スクリプトは**第1週で1本化**。
- **実行環境は WSL2（Ubuntu）に統一。** データはWSLファイルシステム内に置く（/mnt/c は I/O が遅くDataLoaderが詰まる）。リポジトリは手法ごとの uv 独立環境＋直下に共有層（data/ shared/ outputs/）。ストレージは**実用100GB・安全圏150GB**を確保（詳細は共通理解シート §7）。

### 5.1 週次計画
| 週 | テーマ | 全体タスク | 到達点 |
|---|---|---|---|
| **第1週** | 準備・疎通 | 軽量サーベイ／物体データ確定・前処理／3データDL／事前学習済みモデル構築／**スモークテスト**（物体数枚に笑顔出力） | 3手法が物体に出力を出せる＋固定100枚＋2条件の学習集合定義が完成 |
| **第2週** | Baseline生成 | 各手法のBaseline（パレイドリアなし）を100枚×強度3段階で生成。評価スクリプト並行実装 | Baselineの生成と自動指標が揃う |
| **第3週** | +Pareidolia生成＋評価 | Faces in Thingsを足した+Pareidolia条件を生成／両条件で自動評価／**人手2AFC**（自然さ・笑顔・介入選好）／統計 | 2条件の数値と人手ランキングが出揃う |
| **第4週** | 発表準備 | 介入効果の解釈、作例選定、**PowerPoint作成**、リハーサル | スライド完成（予備日込み） |

### 5.2 役割分担
- 個人：M1=VAE / M2=GAN / M3=Diffusion。各自 Baseline・+Pareidolia の両条件を担当。
- 横断（並行・第1週で担当確定）：物体データ＆共通前処理＝1名、評価スクリプト＝1名、人手評価運営＝1名、スライド作図＝1名主導。

### 5.3 週末チェックポイント
- 第1週末：3手法とも物体に笑顔出力が出たか。出ない手法はスコープ縮小（例：Diffusionは InstructPix2Pix の指示編集に絞る）。
- 第2週末：Baselineの自動指標が3手法×強度で出たか。
- 第3週末：+Pareidolia生成と人手評価集計まで終わったか → 第4週はスライドに専念。

> **割り切り：** 4週間では厳密な統計的勝敗より「動かして傾向を掴み、Baseline vs +Pareidolia の差を作例＋数値で語れる」ことを優先。

---

## 6. 主要リスクと対策
1. **自然さの介入効果が出ない/小さい** → それも立派な結果。差分を正直に提示し、なぜか（データ量・ドメイン差）を考察。
2. **物体データの選定でドメインが偏る** → Faces in Thingsの物の種類に寄せ、第1週で固定100枚を吟味。
3. **GANのモード崩壊・構造破壊** → ③物体保存をモニタ、必要ならstructure損失。
4. **VAEのぼやけ** → VAE-GAN化前提。
5. **Diffusionの編集強度↔構造保存** → SDEditノイズ強度・ガイダンス係数をグリッド探索。
6. **自動指標が人手と相関しない** → 第2週の小規模パイロットで相関検証、信頼できる指標を早期選別。

---

## 参考（主要文献の入口）
- Hamilton et al., *Seeing Faces in Things: A Model and Dataset for Pareidolia*, arXiv:2409.16143 (2024)。
- StarGAN v2 / GANimation / AttGAN（表情・属性編集GAN）。
- DiffAE / SDEdit / InstructPix2Pix（拡散編集）。
- β-VAE / VAE-GAN（VAE系・潜在編集）。
- 評価：FID/KID, LPIPS/SSIM, OpenFace(AU), 主観評価（2AFC/MOS）。
