import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.features.linguistic_features import extract_linguistic_features, _ttr, _mtld


def test_ttr_unique():
    tokens = ["a", "b", "c"]
    assert _ttr(tokens) == 1.0


def test_ttr_all_same():
    tokens = ["a", "a", "a"]
    assert _ttr(tokens) == 1 / 3


def test_mtld_short_returns_zero():
    assert _mtld(["a", "b"]) == 0.0


def test_extract_linguistic_features_runs():
    text = "えーと、今日はいい天気ですね。えーと、天気がいいですね。"
    # pos_tagged のモックとして空リストでも動作を確認
    feat = extract_linguistic_features(text, [("今日", "NOUN"), ("天気", "NOUN"), ("いい", "ADJ")])
    assert 0.0 <= feat.filler_ratio <= 1.0
    assert 0.0 <= feat.ttr <= 1.0
