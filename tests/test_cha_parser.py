import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.cha_parser import (
    strip_rtf,
    clean_utterance,
    label_from_diagnosis,
    parse_cha,
)


def test_strip_rtf_passthrough_plain():
    plain = "@Begin\n*PAR:\thello .\n"
    assert strip_rtf(plain) == plain


def test_strip_rtf_unwraps():
    rtf = "{\\rtf1\\ansi\\f0 @UTF8\\\n*PAR:\\thi\\\n}"
    out = strip_rtf(rtf)
    assert "@UTF8" in out
    assert "\\rtf1" not in out


def test_clean_removes_timestamp_bullet():
    assert clean_utterance("do it . \x15120_2732\x15") == "do it."


def test_clean_removes_chat_markup():
    s = "and <I was> [//] &-um I had gone &=ges:downhill . \x151_2\x15"
    assert clean_utterance(s) == "and I was I had gone."


def test_clean_keeps_paren_contents():
    assert clean_utterance("I pushed on (th)em .") == "I pushed on them."


def test_label_control_is_zero():
    assert label_from_diagnosis("Control") == 0


def test_label_missing_is_none():
    # 空/不明の診断はラベル付与不能 -> None（0=健常と混同しない）
    assert label_from_diagnosis("") is None
    assert label_from_diagnosis("   ") is None


def test_label_diagnosis_is_one():
    assert label_from_diagnosis("PPA") == 1
    assert label_from_diagnosis("AD") == 1
    assert label_from_diagnosis("MCI") == 1


def test_parse_real_file():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "raw", "depaul1a.cha"
    )
    if not os.path.exists(path):
        return  # 実データ未配置環境ではスキップ
    rec = parse_cha(path)
    assert rec.id == "depaul1a"
    assert rec.label == 1
    assert rec.diagnosis == "PPA"
    # マークアップが残っていないこと
    for tok in ("\x15", "[//]", "&-", "&=", "+\""):
        assert tok not in rec.text
    # 検査者(INV)の発話が混ざっていないこと（"mhm" は INV のみ）
    assert len(rec.text) > 5000
