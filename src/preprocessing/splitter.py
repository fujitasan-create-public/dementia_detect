"""被験者単位の train/val/test 分割。

DementiaBank は同一被験者が複数セッションを持つ（Delaware の `14-1`, `14-2`,
`14-3` は同一人物）。素朴にファイル単位で分割すると同じ人が train と test の
両方に入り、モデルが「その人の口調」を覚えてリークする。ここでは被験者を最小
単位として分割し、同一被験者のセッションが必ず同じ split に入るようにする。

被験者IDの導出（ファイル名 = dataset.csv の id 列から）:
  - Delaware:  `<数字>-<数字>`  -> `Delaware:<label>:<数字>`
      Delaware は MCI 群と Control 群で番号体系が独立（同じ `15` でも
      MCI:15 と Control:15 は別人。demographics で確認済み）。診断群を分ける
      ため label を被験者キーに含める。同一診断群・同一番号の複数セッションは
      同一被験者として扱う（リーク防止のための保守的グルーピング）。
  - それ以外（Baycrest*, depaul* 等）-> 各ファイルが独立した被験者

使い方:
    python -m src.preprocessing.splitter                 # data/processed/ に出力
    python -m src.preprocessing.splitter --test 0.2 --val 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .data_loader import Dialogue, load_dialogues

_DELAWARE_RE = re.compile(r"^(\d+)-\d+$")


def subject_id(dialogue: Dialogue) -> str:
    """Dialogue から被験者IDを導出する。

    Delaware は診断群ごとに番号が独立するため label で名前空間を分ける。
    """
    m = _DELAWARE_RE.match(dialogue.id)
    if m:
        return f"Delaware:{dialogue.label}:{m.group(1)}"
    return f"single:{dialogue.id}"


def group_by_subject(dialogues: list[Dialogue]) -> dict[str, list[Dialogue]]:
    groups: dict[str, list[Dialogue]] = defaultdict(list)
    for d in dialogues:
        groups[subject_id(d)].append(d)
    return dict(groups)


def subject_labels(dialogues: list[Dialogue]) -> dict[str, int]:
    """被験者ごとのラベル（セッション多数決）。ラベル不一致は警告付きで多数決。"""
    labels: dict[str, int] = {}
    for subj, items in group_by_subject(dialogues).items():
        counts = Counter(d.label for d in items)
        if len(counts) > 1:
            print(f"[warn] subject {subj} has mixed labels {dict(counts)}; "
                  f"using majority")
        labels[subj] = counts.most_common(1)[0][0]
    return labels


def split_by_subject(
    dialogues: list[Dialogue],
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> tuple[list[Dialogue], list[Dialogue], list[Dialogue]]:
    """被験者単位・ラベル層化で train/val/test に分割する。

    同一被験者の全セッションは必ず同じ split に入る（リーク防止）。
    層化は被験者ラベル単位で行うため、split 間のクラス比がほぼ揃う。
    戻り値は Dialogue のリスト 3 つ (train, val, test)。
    """
    labels = subject_labels(dialogues)
    subjects_by_label: dict[int, list[str]] = defaultdict(list)
    for subj, lab in labels.items():
        subjects_by_label[lab].append(subj)

    rng = random.Random(seed)
    assign: dict[str, str] = {}
    for lab, subs in subjects_by_label.items():
        subs = sorted(subs)          # 決定性のため一旦ソート
        rng.shuffle(subs)
        n = len(subs)
        n_test = round(n * test_size)
        n_val = round(n * val_size)
        for s in subs[:n_test]:
            assign[s] = "test"
        for s in subs[n_test:n_test + n_val]:
            assign[s] = "val"
        for s in subs[n_test + n_val:]:
            assign[s] = "train"

    buckets: dict[str, list[Dialogue]] = {"train": [], "val": [], "test": []}
    for d in dialogues:
        buckets[assign[subject_id(d)]].append(d)
    return buckets["train"], buckets["val"], buckets["test"]


def _write_csv(dialogues: list[Dialogue], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "subject", "text", "label"])
        for d in dialogues:
            w.writerow([d.id, subject_id(d), d.text, d.label])


def _report(name: str, dialogues: list[Dialogue]) -> None:
    subs = {subject_id(d) for d in dialogues}
    lab = Counter(d.label for d in dialogues)
    print(f"  {name:5s}: {len(dialogues):3d} sessions / {len(subs):3d} subjects "
          f"| label0={lab[0]} label1={lab[1]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="被験者単位 train/val/test 分割")
    ap.add_argument("--source", default="raw", help="DATA_SOURCE (default raw)")
    ap.add_argument("--out", default="data/processed", help="出力ディレクトリ")
    ap.add_argument("--test", type=float, default=0.2, help="test 割合（被験者比）")
    ap.add_argument("--val", type=float, default=0.1, help="val 割合（被験者比）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dialogues = load_dialogues(args.source)
    train, val, test = split_by_subject(dialogues, args.test, args.val, args.seed)

    out = Path(args.out)
    _write_csv(train, out / "train.csv")
    _write_csv(val, out / "val.csv")
    _write_csv(test, out / "test.csv")

    # リーク検査
    tr = {subject_id(d) for d in train}
    va = {subject_id(d) for d in val}
    te = {subject_id(d) for d in test}
    overlap = (tr & va) | (tr & te) | (va & te)
    print(f"total: {len(dialogues)} sessions / {len(tr | va | te)} subjects")
    _report("train", train)
    _report("val", val)
    _report("test", test)
    print(f"subject overlap between splits: {len(overlap)} "
          f"({'OK' if not overlap else 'LEAK!'})")
    print(f"-> {out}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
