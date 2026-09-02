"""spaCy en_core_web_md を使った英語形態素解析ラッパー。

モデル未導入なら:  python -m spacy download en_core_web_md
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy


_nlp = None
_MODEL = "en_core_web_md"


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load(_MODEL)
    return _nlp


def tokenize(text: str) -> list[str]:
    """テキストをトークンリストに変換する。ストップワード・空白を除去済み。"""
    doc = _get_nlp()(text)
    return [t.lemma_ for t in doc if not t.is_stop and not t.is_space]


def pos_tag(text: str) -> list[tuple[str, str]]:
    """(lemma, pos) のリストを返す"""
    doc = _get_nlp()(text)
    return [(t.lemma_, t.pos_) for t in doc if not t.is_space]
