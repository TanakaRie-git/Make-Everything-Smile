"""FacesInThings（パレイドリア＝物体に見える人間の顔）データセットのローダ。

推論時に「物の中の顔」を入力として笑顔化するために使う。
公式: https://github.com/mhamilton723/FacesInThings

公式パッケージ（pip install facesinthings）は pandas に依存するが、ここでは
diffusion 環境に余計な依存を足さないよう **stdlib の csv だけ** で metadata を読む。

ダウンロード仕様（公式 dataset.py と同じ）:
    https://aka.ms/faces-dataset  →  FacesInThings.zip
    展開すると <root>/FacesInThings/ 以下に
        images/         個別 jpg（例: 000000009.jpg）
        metadata.csv    アノテーション

metadata.csv の主な列:
    file        画像ファイル名（images/ 配下）
    boxes       顔の矩形 [x1, y1, w, h] のリストを JSON 文字列にしたもの
    is_primary  主たる顔か
    num_boxes   矩形数
    train       学習/テスト split フラグ（True/False 系の文字列）
    その他       Emotion? / Gender? / Hard to spot? などの知覚アノテーション

このモジュールは PyTorch に依存しない軽量ローダ（推論スクリプトから使う）。
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from PIL import Image

DATASET_URL = "https://aka.ms/faces-dataset"


@dataclass
class FaceSample:
    """metadata.csv の 1 行 ＋ 画像パス。"""

    path: Path
    file: str
    boxes: list[list[float]]          # [[x1, y1, w, h], ...]
    is_primary: bool
    train: bool
    meta: dict[str, str] = field(default_factory=dict)  # 生の行（その他の列）

    def open_image(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")

    def primary_box(self) -> list[float] | None:
        """代表となる顔矩形を返す（無ければ None）。"""
        if not self.boxes:
            return None
        # 面積最大の矩形を顔とみなす（is_primary は画像単位なので矩形選択には使えない）。
        return max(self.boxes, key=lambda b: b[2] * b[3])


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def download_dataset(root: str | Path) -> Path:
    """データセットを <root>/FacesInThings/ に用意する（無ければDL・展開）。"""
    import requests

    root = Path(root)
    dataset_dir = root / "FacesInThings"
    if dataset_dir.exists():
        return dataset_dir

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "FacesInThings.zip"
    print(f"[facesinthings] dataset が無いのでダウンロードします: {DATASET_URL}")
    with requests.get(DATASET_URL, stream=True) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"[facesinthings] 展開中: {zip_path} -> {root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return dataset_dir


class FacesInThings:
    """FacesInThings データセットの軽量イテレータ。

    Args:
        root: データセットの親ディレクトリ。<root>/FacesInThings/ を探す。
        download: 見つからなければ DATASET_URL からDLする。
        split: "all" | "train" | "test"。metadata の train 列でフィルタ。
        primary_only: is_primary な画像だけに絞る。
    """

    def __init__(
        self,
        root: str | Path,
        download: bool = False,
        split: str = "all",
        primary_only: bool = False,
    ):
        root = Path(root)
        dataset_dir = root / "FacesInThings"
        if not dataset_dir.exists():
            if download:
                dataset_dir = download_dataset(root)
            else:
                raise FileNotFoundError(
                    f"FacesInThings が見つかりません: {dataset_dir}\n"
                    f"--download を付けるか、{DATASET_URL} を手動で展開してください。"
                )

        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.metadata_file = dataset_dir / "metadata.csv"
        if not self.metadata_file.is_file():
            raise FileNotFoundError(f"metadata.csv が見つかりません: {self.metadata_file}")

        self.samples = self._load_metadata(split=split, primary_only=primary_only)
        if not self.samples:
            raise ValueError(
                f"条件に合うサンプルがありません (split={split}, primary_only={primary_only})"
            )

    def _load_metadata(self, split: str, primary_only: bool) -> list[FaceSample]:
        samples: list[FaceSample] = []
        with self.metadata_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    boxes = json.loads(row.get("boxes") or "[]")
                except (json.JSONDecodeError, TypeError):
                    boxes = []
                is_primary = _as_bool(row.get("is_primary"))
                is_train = _as_bool(row.get("train"))

                if split == "train" and not is_train:
                    continue
                if split == "test" and is_train:
                    continue
                if primary_only and not is_primary:
                    continue

                samples.append(
                    FaceSample(
                        path=self.images_dir / row["file"],
                        file=row["file"],
                        boxes=boxes,
                        is_primary=is_primary,
                        train=is_train,
                        meta=row,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FaceSample:
        return self.samples[idx]

    def __iter__(self) -> Iterator[FaceSample]:
        return iter(self.samples)
