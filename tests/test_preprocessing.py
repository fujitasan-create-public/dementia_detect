import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.text_normalizer import normalize


def test_normalize_lowercases():
    assert normalize("Hello WORLD") == "hello world"


def test_normalize_removes_symbols():
    result = normalize("great!! text 🎉")
    assert "🎉" not in result
    assert "!" not in result


def test_normalize_keeps_intra_word_apostrophe():
    assert normalize("I don't know") == "i don't know"


def test_normalize_whitespace():
    result = normalize("a   b\t\tc")
    assert result == "a b c"
