import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.text_normalizer import normalize


def test_normalize_removes_symbols():
    result = normalize("テスト！！　text🎉")
    assert "🎉" not in result
    assert "！" not in result


def test_normalize_unicode():
    result = normalize("ｱｲｳ")  # 半角カナ -> 全角
    assert "ア" in result


def test_normalize_whitespace():
    result = normalize("a   b\t\tc")
    assert result == "a b c"
