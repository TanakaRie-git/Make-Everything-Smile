# Claude Design 依頼書 — "Make Everything Smile" 発表スライド

VAE / GAN / Diffusion による「物体への笑顔付与」比較発表のスライドデッキを作成してください。

## 全体仕様

| 項目 | 内容 |
|---|---|
| 発表時間 | **10分**（本編14枚 + 予備3枚。1枚あたり約40秒、山場のS9のみ100秒） |
| 言語 | スライド本文は**英語**、この依頼書の指示は日本語 |
| トーン | 研究発表だが遊び心のあるテーマ。クリーンで写真（作例画像）が主役。文字は少なく |
| 画像パス | すべてリポジトリルートからの相対パス（`vae/assets/...`）。**画像は切り抜き・再圧縮せずそのまま配置**（比較ストリップは列の並びに意味がある） |
| 未確定要素 | GAN パート（S6・S10）と Diffusion の作例（S11）、比較表の GAN/Diffusion 列（S12）は**プレースホルダ**とし、差し替え可能なスロットにする |

## 全スライド共通のデザイン要素

1. **「input → inject → output」の共通図**を S4 で定義し、S5 / S6 / S7 で再掲して**注入点だけをハイライト**する（3手法の違いが「どこに注入するか」だけ、という主張を視覚で通す）。
2. 比較ストリップ画像（`cmp_*.jpg`）は横並びの連続フレーム。**列ラベル**（`input | α=0 | α=1 | α=2 | α=3` など）を画像の上か下に小さく付ける。
3. メソッドカラー: VAE / GAN / Diffusion に一貫した3色を割り当て、手法スライド・結果スライド・比較表で使い回す。

---

## S1 — Title

**MAKE EVERYTHING SMILE**

# Make everyday objects smile.

Adding a "smile" with VAE / GAN / Diffusion

Tags: `VAE` · `GAN` · `Diffusion`

- 背景またはアクセントに笑顔化した物体のキービジュアル [vae/assets/current/cmp_object_smile_keyvis.jpg](../assets/current/cmp_object_smile_keyvis.jpg) の最終フレーム（右端の笑ったワッフル）を使ってよい。

---

## S2 — Motivation & Background（旧02+03を統合）

**MOTIVATION · PAREIDOLIA**

### Why make things smile?

1. Making objects smile is **fun — and almost unstudied**.
2. Humans already see faces in objects: **pareidolia**.
3. A dataset of face-like objects exists — **Faces in Things** (Hamilton et al., ECCV 2024) — our foothold.

- 出典クレジット1行: *Faces in Things: Hamilton et al., arXiv:2409.16143*。
- パレイドリアの説明用に Faces in Things 系の入力画像を1〜2枚: [vae/assets/current/input_bun_000007868.jpg](../assets/current/input_bun_000007868.jpg)（「これが顔に見えますよね？」）。

---

## S3 — Goal, Task & Evaluation（旧04+05を統合）

**GOAL & EVALUATION**

### Make the same object smile

- **INPUT:** face-like object crops from Faces in Things (**10 shared IDs, all methods**)
- **Smile source:** two domains — CelebA-HQ "Smiling" (human) & Faces in Things neutral→happy (object faces)
- **OUTPUT:** the same object, now smiling

The hard parts: **no ground-truth pairs**, and **preserve the original object**.

### Three ways to judge success

1. **Preserved?** — object not broken (LPIPS · SSIM · CLIP image sim)
2. **Did it smile?** — target expression added (automatic metrics *and their limits* → S12)
3. **Natural?** — no artifacts (human judgement)

- 注意: 解像度は書かない（手法間で 256px / 512px と不統一のため）。「共通の10 crop」が土俵の共通性の主張。
- レイアウト: 上半分にタスク（input→output の絵）、下半分に評価3軸。

---

## S4 — Methods Overview

**METHODS · OVERVIEW**

### Only the injection point differs

INPUT (object) → **inject a smile** → OUTPUT (same object, smiling)

- **VAE:** add a direction vector in latent space — `z′ = z + α·d` · **no training**
- **GAN:** give it as a condition at generation — expression label · adversarial
- **Diffusion:** edit with a text instruction — "make the face smile" + LoRA

Every method is "input → inject → output." Only the **place and manner of injection** differ.

- ここで共通図（input → inject → output）を定義。以降の手法スライドで再掲・ハイライト。
- 変更点（旧06から）: VAE の「trained once」→ **no training**（凍結SD-VAE＋方向1本のみ、が実装事実）。

---

## S5 — Method · VAE

**METHOD · VAE**

### A "smile direction" in latent space

```
d  = normalize( mean(z_smile) − mean(z_neutral) )
z′ = z + α · σ · d        (σ: projection std — α is comparable across directions)
```

Encode with a **frozen, pretrained SD-VAE** — the *same* image-drawing VAE family Diffusion uses. **No training**: just one direction vector. Strength is a continuous dial **α**.

Control is complete with one vector. Quality & preservation depend on the **backbone (decoder)**.

- 画像（右下に小さく）: [vae/assets/old/v3_faces_sanity.jpg](../assets/old/v3_faces_sanity.jpg)
  キャプション: *sanity check — the direction does produce smiles on faces (64px scratch model)*
- 「same backbone family as Diffusion」の一文は S9 への伏線なので視覚的に残す（下線 or 色）。

---

## S6 — Method · GAN（プレースホルダ）

**METHOD · GAN**

### Conditional generation

*To be written by the GAN team* — スロット構成のみ用意:

- Method（1図）
- Examples（画像スロット）
- Findings（1行）

---

## S7 — Method · Diffusion

**METHOD · DIFFUSION**

### Editing with words

- Built on a pretrained **InstructPix2Pix**. VAE, text encoder and U-Net are **frozen**; only a small **LoRA (rank 8)** on attention is trained.
- Trained on **Faces in Things (neutral→happy)**. Inference: "make the face smile", conditioned on the input image (50 steps · image guidance ≈ 1.5).
- The image-drawing VAE is **pretrained & frozen** — sharp, with strong preservation.

Pipeline: Input image + instruction → Pretrained IP2P (frozen) + LoRA (trained) → Smiling output (object preserved)

- 「pretrained & frozen」を強調表示（S5 と同じ装飾）。S9 で「VAE も同じ土台に載せた」と回収する。
- Diffusion 側の作例画像は Diffusion チームから受領（スロット）。

---

## S8 — Experimental Setup

**EXPERIMENTAL SETUP**

### Change only the intervention

Same data and evaluation for every method — only the way the smile is injected differs.

- **DATASET — CelebA-HQ:** human faces (smiling / neutral). *Baseline* smile direction.
- **DATASET — Faces in Things:** face-like objects (neutral→happy). Main smile source **and** the evaluation inputs.
- **EVALUATION:** 10 shared Faces-in-Things crops (`eval_ids.txt`) · identical output format for all methods · metrics + human eval.

> Faces smile easily with every method — **the difference shows on objects.** (→ next)

- 旧10+11 の統合。顔での結果スライドは作らず、末尾の1行（引用枠）で処理する。

---

## S9 — Results · Objects (VAE) 【山場・100秒】

**RESULTS · OBJECTS (VAE)**

### The weakness is the backbone, not VAE

**同一物体（蒸しパン, ID 000007868）の3段変化**を横に並べる。これがこのスライドの核。

| 段 | 画像 | ラベル（英語） |
|---|---|---|
| 入力 | [vae/assets/current/input_bun_000007868.jpg](../assets/current/input_bun_000007868.jpg) | INPUT |
| ① | [vae/assets/current/cmp_scratch_humanize_bun.jpg](../assets/current/cmp_scratch_humanize_bun.jpg) | **Scratch VAE-GAN** — works, but blurry & turns *human* |
| ② | [vae/assets/current/cmp_pretrained_sharp_bun.jpg](../assets/current/cmp_pretrained_sharp_bun.jpg) | **Same math, pretrained SD-VAE backbone** — sharp at α=0 |
| ③ | [vae/assets/current/cmp_object_smile_bun.jpg](../assets/current/cmp_object_smile_bun.jpg) | **FiT direction + color-keep** — the object's *own* face smiles |

下段にキービジュアルを大きく: [vae/assets/current/cmp_object_smile_keyvis.jpg](../assets/current/cmp_object_smile_keyvis.jpg)
キャプション: *Original color & shape kept — the waffle smiles with its own face.*

数値の一言（①→③、共通10枚）: **LPIPS 0.74 → 0.17 · SSIM 0.35 → 0.79**

余白があれば小さく2枚: [cmp_object_smile_waffle.jpg](../assets/current/cmp_object_smile_waffle.jpg) / [cmp_object_smile_potato.jpg](../assets/current/cmp_object_smile_potato.jpg)（入らなければ予備スライドへ）。

- レイアウト優先順: ①②③の3段比較 ＞ キービジュアル ＞ 数値 ＞ 追加作例。
- ①の画像では α=0 列（左から2列目）が既にボケていることを矢印等で指す。
- ②では α=0 列がシャープなことを同じ装飾で対比。

---

## S10 — Results · Objects (GAN)（プレースホルダ）

**RESULTS · OBJECTS (GAN)**

Input row / Output row — 画像スロットのみ（GAN チームが差し込み）。

---

## S11 — Results · Objects (Diffusion)（プレースホルダ）

**RESULTS · OBJECTS (DIFFUSION)**

Input row / Output row — 画像スロットのみ（Diffusion チームが差し込み）。

---

## S12 — Results · Metrics

**RESULTS · METRICS**

### Preservation is measurable — "smile" is not (yet)

主表（実測値、共通10枚。GAN / Diffusion 列は同一スクリプト `vae/experiments/eval.py` で追記予定のスロット）:

| Preservation (10 shared crops) | scratch VAE-GAN | **SD-VAE + FiT dir + color-keep** | GAN | Diffusion |
|---|---|---|---|---|
| LPIPS ↓ | 0.74 | **0.17** | *TBD* | *TBD* |
| SSIM ↑ | 0.35 | **0.79** | *TBD* | *TBD* |
| CLIP image sim ↑ | 0.67 | **0.92** | *TBD* | *TBD* |

脚注（結果として提示する発見）:

> **Automatic smile metrics failed:** CLIP-smile Δ ≈ ±0.01 (noise) — it only rewards outputs that turn *human*. Judging "did the object smile?" needs **human evaluation**.

- Naturalness 行は入れない（FID/KID は10枚では計算不能。人手評価は今後）。
- 太字列（SD-VAE版）にメソッドカラー（VAE色）を敷く。

---

## S13 — Discussion

**DISCUSSION**

### What actually made the difference

1. **The backbone, not the method family.** The visible gap between methods is dominated by *having a pretrained image-drawing backbone*. Put VAE on the same frozen SD-VAE and it becomes sharp too.
2. **The remaining, essential difference:** VAE = one **linear latent vector** (no training, light, transparent). Diffusion = a **learned, image-conditioned edit** (precise, heavy). Color drift & fixed smile position in VAE are symptoms of this gap.
3. **Evaluation trap:** automatic smile metrics reward *humanization*. Object-preserving smiles need human judgement.

- 3点を番号付きで大きく。図は不要（文字スライドはここだけ許容）。

---

## S14 — Finally · One more thing

**FINALLY · ONE MORE THING**

### ChatGPT does it with words alone.

- house: original → smiling
- block: original → smiling（画像はチーム提供スロット）

An ordinary object smiles from a single sentence. **— Why can it?**

**The same answer as today: a giant pretrained backbone.** / Next time, we'll get the house and the road to smile too.

- S13 の結論1をそのまま回収する構成。「今日の結論そのものです」のトーンで。

---

## 予備スライド（質疑用・3枚）

### B1 — VAE: All 10 IDs（チェリーピッキングでない証拠）
- [vae/assets/results/sdvae_fit_kc_overview.jpg](../assets/results/sdvae_fit_kc_overview.jpg)（現行手法・全ID）
- [vae/assets/results/scratch_baseline_overview.jpg](../assets/results/scratch_baseline_overview.jpg)（スクラッチ版・全ID、対比用）

### B2 — VAE: Smile-strength ↔ preservation trade-off
- [vae/assets/results/sdvae_cascade_overview.jpg](../assets/results/sdvae_cascade_overview.jpg)
- メッセージ: *Cascading makes strong toothy smiles — but the object turns human. Two poles: keep the object ↔ maximize the smile.*

### B3 — VAE: Why the smile lands at a fixed position
- [vae/assets/results/sdvae_low_celeba_kc_ALIGNED_overview.jpg](../assets/results/sdvae_low_celeba_kc_ALIGNED_overview.jpg)
- メッセージ: *Aligning inputs helps, but a linear latent direction cannot track each object's mouth — a principled limit of linear editing (vs Diffusion's image-conditioned edit).*

---

## 画像アセット早見表（VAE 分・すべて git 管理済み）

| 用途 | パス |
|---|---|
| 入力（蒸しパン） | `vae/assets/current/input_bun_000007868.jpg` |
| スクラッチ=人間化 | `vae/assets/current/cmp_scratch_humanize_bun.jpg` |
| SD-VAE=シャープ | `vae/assets/current/cmp_pretrained_sharp_bun.jpg` |
| 物体自身が笑う | `vae/assets/current/cmp_object_smile_bun.jpg` |
| キービジュアル（ワッフル） | `vae/assets/current/cmp_object_smile_keyvis.jpg` |
| 追加作例 | `vae/assets/current/cmp_object_smile_waffle.jpg`, `cmp_object_smile_potato.jpg` |
| 顔サニティ | `vae/assets/old/v3_faces_sanity.jpg` |
| 予備: 全ID俯瞰ほか | `vae/assets/results/*.jpg` |

## 未確定・差し替え予定（スロット化しておく箇所）

1. S6 / S10: GAN パート一式（GAN チーム提供待ち）
2. S11: Diffusion 作例（Diffusion チーム提供待ち）
3. S12: GAN / Diffusion 列の数値（`vae/experiments/eval.py` を各出力に実行して確定）
4. S14: ChatGPT 作例画像（house / block）
