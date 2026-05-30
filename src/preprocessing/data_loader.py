"""対話データのロード。DATA_SOURCE 環境変数で使うデータを切り替える。

DATA_SOURCE=raw       -> data/raw/       (DementiaBank 到着後)
DATA_SOURCE=translated -> data/translated/ (デフォルト: 翻訳データ)
DATA_SOURCE=synthetic  -> data/synthetic/
"""
import os
import csv
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Dialogue:
    id: str
    text: str          # 発話テキスト（前処理済み or 生）
    label: int         # 1=認知機能低下あり, 0=なし
    source: str        # "raw" | "translated" | "synthetic"


_BASE = Path(__file__).resolve().parents[2] / "data"
_SOURCE_MAP = {
    "raw": _BASE / "raw",
    "translated": _BASE / "translated",
    "synthetic": _BASE / "synthetic",
}


def load_dialogues(source: str | None = None) -> list[Dialogue]:
    """CSV ファイルから対話リストを返す。

    CSV フォーマット: id,text,label
    source を省略すると環境変数 DATA_SOURCE を参照し、なければ translated を使う。
    """
    source = source or os.getenv("DATA_SOURCE", "translated")
    data_dir = _SOURCE_MAP[source]
    dialogues = []
    for csv_path in sorted(data_dir.glob("*.csv")):
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogues.append(Dialogue(
                    id=row["id"],
                    text=row["text"],
                    label=int(row["label"]),
                    source=source,
                ))
    return dialogues
