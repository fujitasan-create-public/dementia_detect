import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.data_loader import Dialogue
from src.preprocessing.splitter import (
    subject_id,
    group_by_subject,
    split_by_subject,
)


def _d(id_, label):
    return Dialogue(id=id_, text="x " * 300, label=label, source="raw")


def test_subject_id_delaware_namespaced_by_label():
    # 同じ番号でも診断群が違えば別被験者
    assert subject_id(_d("15-1", 1)) == "Delaware:1:15"
    assert subject_id(_d("15-3", 0)) == "Delaware:0:15"
    assert subject_id(_d("15-1", 1)) != subject_id(_d("15-3", 0))


def test_subject_id_sessions_grouped():
    # 同一診断群・同一番号の複数セッションは同じ被験者
    assert subject_id(_d("14-1", 1)) == subject_id(_d("14-2", 1)) == "Delaware:1:14"


def test_subject_id_single_files():
    assert subject_id(_d("Baycrest8961", 1)) == "single:Baycrest8961"
    assert subject_id(_d("depaul1a", 1)) == "single:depaul1a"


def test_group_by_subject_counts():
    ds = [_d("14-1", 1), _d("14-2", 1), _d("14-3", 0), _d("Baycrest1", 1)]
    groups = group_by_subject(ds)
    assert set(groups) == {"Delaware:1:14", "Delaware:0:14", "single:Baycrest1"}
    assert len(groups["Delaware:1:14"]) == 2


def _make_dataset():
    ds = []
    for n in range(1, 41):          # 40 MCI subjects, 各2セッション
        ds += [_d(f"{n}-1", 1), _d(f"{n}-2", 1)]
    for n in range(100, 160):       # 60 Control subjects, 各1セッション
        ds.append(_d(f"{n}-1", 0))
    return ds


def test_no_subject_leakage():
    train, val, test = split_by_subject(_make_dataset(), seed=42)
    s = lambda xs: {subject_id(d) for d in xs}
    tr, va, te = s(train), s(val), s(test)
    assert tr & va == set()
    assert tr & te == set()
    assert va & te == set()


def test_all_sessions_assigned_once():
    ds = _make_dataset()
    train, val, test = split_by_subject(ds, seed=42)
    ids = [d.id for d in train] + [d.id for d in val] + [d.id for d in test]
    assert sorted(ids) == sorted(d.id for d in ds)
    assert len(ids) == len(set(ids))


def test_deterministic():
    ds = _make_dataset()
    a = split_by_subject(ds, seed=42)
    b = split_by_subject(ds, seed=42)
    assert [d.id for d in a[2]] == [d.id for d in b[2]]
    c = split_by_subject(ds, seed=7)
    # 別シードでは test 集合が変わる（極稀な一致を避けるため集合比較）
    assert {d.id for d in a[2]} != {d.id for d in c[2]}


def test_stratification_roughly_holds():
    ds = _make_dataset()
    train, val, test = split_by_subject(ds, test_size=0.2, val_size=0.1, seed=42)
    def pos_ratio(xs):
        return sum(d.label for d in xs) / len(xs)
    overall = pos_ratio(ds)
    for split in (train, test):
        assert abs(pos_ratio(split) - overall) < 0.15
