"""DementiaBank CHAT (.cha) 形式 -> CSV(id,text,label) 変換。

DementiaBank の生データは .cha (CHAT) 形式で配布される。GUI からコピペ保存
すると TextEdit などが RTF で包んでしまう（各行末に `\\`、先頭に `{\\rtf1...`）
ため、RTF 除去も込みで処理する。

抽出方針（一旦英語のまま。翻訳は後段パイプラインで噛ませる）:
  - 話者 `*PAR:`（被験者）の発話のみを連結して 1 サンプル = 1 対話とする。
  - `%mor` / `%gra` 等の依存層、`*INV:`（検査者）発話は捨てる。
  - CHAT のアノテーション記号（[//], &-um, タイムスタンプ等）を除去し、
    素の発話テキストに戻す。
  - ラベルは `@ID` 行の diagnosis フィールドから決定
    （Control/空 -> 0、それ以外の診断名 -> 1）。

使い方:
    python -m src.preprocessing.cha_parser data/raw/*.cha -o data/raw/dataset.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


# --- RTF 除去 -----------------------------------------------------------------

def strip_rtf(text: str) -> str:
    """コピペ保存で付く RTF ラッピングを取り除いて素の .cha 本文を返す。

    RTF でなければそのまま返す。RTF の場合は制御ワード・波括弧・
    行継続の `\\` を除去して復元する。
    """
    if not text.lstrip().startswith("{\\rtf"):
        return text

    # RTF ヘッダ（フォント表・カラーテーブル等）をざっくり落とす
    body = text
    # 制御ワード（\rtf1, \ansi, \f0, \fs26 ...）を除去
    body = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", body)
    # エスケープされた波括弧を退避
    body = body.replace(r"\{", "\x01").replace(r"\}", "\x02")
    # 生の波括弧（グループ区切り）を除去
    body = body.replace("{", "").replace("}", "")
    body = body.replace("\x01", "{").replace("\x02", "}")
    # RTF では元の改行が `\` + 改行 で表現される。まず `\`+改行 -> 改行
    body = re.sub(r"\\\r?\n", "\n", body)
    # 残った行末 `\`（改行前）も改行として扱う
    body = body.replace("\\\n", "\n")
    return body


# --- CHAT アノテーション除去 --------------------------------------------------

def clean_utterance(line: str) -> str:
    """1 発話行から CHAT のマークアップを剥がして素のテキストにする。"""
    s = line
    # CHAT のタイムスタンプは NAK(0x15) で挟まれる  例: "\x15120_2732\x15"
    s = re.sub(r"\x15[^\x15]*\x15", "", s)
    # NAK に挟まれない裸のタイムスタンプ（行末）も念のため除去
    s = re.sub(r"\s*\d+_\d+\s*$", "", s)
    # 各種置換・エラー・ジェスチャ注記   [//] [/] [: x] [* p:w] [=! ...] [%...]
    s = re.sub(r"\[[^\]]*\]", "", s)
    # ジェスチャ/イベント &=ges:downhill, &=points:ribs, &=ges
    s = re.sub(r"&=\S+", "", s)
    # フィラー・言い淀み &-um, &+th, &+ha, &*INV:mhm 等の & 系
    s = re.sub(r"&[-+*]\S+", "", s)
    # 話者内注記マーカー  +"/. +//. +< +" など
    s = re.sub(r'\+["/<]+\S*', "", s)
    s = re.sub(r"\+//?\.", "", s)
    # <...> のスコープ括弧（中身は残す）
    s = s.replace("<", "").replace(">", "")
    # (th)em のような省略括弧 -> 中身を残す
    s = re.sub(r"\(([^)]*)\)", r"\1", s)
    # 判別不能トークン xxx, yyy, www
    s = re.sub(r"\b[xyw]{3}\b", "", s)
    # 単独の & や記号の掃除
    s = s.replace("&", " ")
    # 句読点前の余分な空白を整え、連続空白を 1 個に
    s = re.sub(r"\s+([.?!,])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --- ラベル判定 ---------------------------------------------------------------

_CONTROL_TOKENS = {"control", "hc", "healthy", "normal"}


def label_from_diagnosis(diagnosis: str) -> int | None:
    """@ID の diagnosis フィールドからラベルを決める。

    Control -> 0（低下なし）、診断名（PPA, AD, MCI...） -> 1。
    空文字・不明（@ID 欠損等）-> None（ラベル付与不能。データセットから除外）。
    """
    d = diagnosis.strip().lower()
    if not d:
        return None
    return 0 if d in _CONTROL_TOKENS else 1


# --- パース本体 ---------------------------------------------------------------

@dataclass
class ChaRecord:
    id: str
    text: str
    label: int | None   # 1=低下, 0=健常, None=ラベル不能（除外対象）
    diagnosis: str


def parse_cha(path: str | Path) -> ChaRecord:
    """1 つの .cha ファイルを 1 レコードにパースする。"""
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    content = strip_rtf(raw)

    diagnosis = ""
    utterances: list[str] = []
    speaker_re = re.compile(r"^\*(\w+):\t?(.*)$")

    # CHAT は 1 発話がタブ折り返しで複数行になることがあるので継続行を結合
    lines = content.splitlines()
    current: str | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            cleaned = clean_utterance(current)
            if cleaned:
                utterances.append(cleaned)
            current = None

    for line in lines:
        # @ID: eng|DePaul|PAR|66;00.|female|PPA||Participant|||
        if line.startswith("@ID:"):
            fields = line.split("\t", 1)[-1].split("|")
            # fields: [lang, corpus, speaker, age, sex, group(=diagnosis), ...]
            if len(fields) > 5 and fields[2].strip() == "PAR":
                diagnosis = fields[5]
            continue
        m = speaker_re.match(line)
        if m:
            flush()
            speaker, utt = m.group(1), m.group(2)
            current = utt if speaker == "PAR" else None
        elif line.startswith("%") or line.startswith("@"):
            # 依存層 (%mor,%gra) やヘッダは発話継続を打ち切る
            flush()
        elif current is not None and not line.startswith("*"):
            # タブ継続行（前の *PAR 発話の続き）
            current += " " + line.strip()
    flush()

    text = " ".join(utterances)
    text = re.sub(r"\s+", " ", text).strip()
    return ChaRecord(
        id=path.stem,
        text=text,
        label=label_from_diagnosis(diagnosis),
        diagnosis=diagnosis.strip(),
    )


def write_csv(records: list[ChaRecord], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "label"])
        for r in records:
            writer.writerow([r.id, r.text, r.label])


def is_usable(record: ChaRecord, min_chars: int = 200) -> bool:
    """データセットに採用可能か。ラベル不能・空/極端に短い書き起こしを除外する。"""
    return record.label is not None and len(record.text) >= min_chars


def main() -> None:
    ap = argparse.ArgumentParser(description=".cha -> CSV(id,text,label) 変換")
    ap.add_argument("inputs", nargs="+", help=".cha ファイル / ディレクトリ（複数可・glob 可）")
    ap.add_argument("-o", "--out", default="data/raw/dataset.csv", help="出力 CSV")
    ap.add_argument("--min-chars", type=int, default=200,
                    help="この文字数未満の書き起こしは除外（デフォルト 200）")
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.inputs:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(p.rglob("*.cha"))
        elif any(c in pattern for c in "*?["):
            paths.extend(Path().glob(pattern))
        else:
            paths.append(p)

    records = [parse_cha(p) for p in sorted(set(paths))]
    usable = [r for r in records if is_usable(r, args.min_chars)]
    excluded = [r for r in records if not is_usable(r, args.min_chars)]

    write_csv(usable, args.out)

    n0 = sum(1 for r in usable if r.label == 0)
    n1 = sum(1 for r in usable if r.label == 1)
    print(f"-> {args.out}: {len(usable)} records  (label0={n0}, label1={n1})")
    if excluded:
        print(f"excluded {len(excluded)}:")
        for r in excluded:
            reason = "no-label" if r.label is None else f"too-short({len(r.text)}c)"
            print(f"  {r.id}: {reason}")


if __name__ == "__main__":
    main()
