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
    text = "um the weather is nice today. um the weather is nice."
    feat = extract_linguistic_features(
        text, [("weather", "NOUN"), ("nice", "ADJ"), ("today", "NOUN")]
    )
    assert 0.0 <= feat.filler_ratio <= 1.0
    assert 0.0 <= feat.ttr <= 1.0


def test_english_fillers_detected():
    text = "um well I mean you know it's like whatever."
    feat = extract_linguistic_features(text, [("mean", "VERB")])
    assert feat.filler_ratio > 0.0
