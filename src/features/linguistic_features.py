"""ルールベースの表層言語特徴量（TTR, MTLD, POS分布, フィラー頻度等）。

LLMCARE 論文の 110 特徴から英語で計算可能なものを実装。
"""
from __future__ import annotations
import re
import math
from dataclasses import dataclass, fields


# 英語フィラー / ディスフルエンシー・ディスコースマーカー
_FILLER_PATTERN = re.compile(
    r"\b(uh|uhm|um+|erm?|hmm+|mhm|uh[- ]?huh|ah|"
    r"well|like|you know|y'know|i mean|sort of|kind of|"
    r"actually|basically|literally|whatever)\b",
    re.IGNORECASE,
)


@dataclass
class LinguisticFeatures:
    ttr: float = 0.0           # Type-Token Ratio
    mtld: float = 0.0          # Measure of Textual Lexical Diversity
    avg_sentence_len: float = 0.0
    noun_ratio: float = 0.0
    verb_ratio: float = 0.0
    adj_ratio: float = 0.0
    filler_ratio: float = 0.0  # フィラー数 / 総トークン数
    repetition_ratio: float = 0.0  # 繰り返しフレーズの割合

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _ttr(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """MTLD: TTR が threshold を下回るまでのトークン数の平均。"""
    if len(tokens) < 10:
        return 0.0

    def _factor_count(toks: list[str]) -> float:
        types: set[str] = set()
        token_count = 0
        factors = 0.0
        for t in toks:
            types.add(t)
            token_count += 1
            current_ttr = len(types) / token_count
            if current_ttr <= threshold:
                factors += 1
                types = set()
                token_count = 0
        if token_count > 0:
            factors += (1.0 - len(types) / token_count) / (1.0 - threshold)
        return factors

    fwd = _factor_count(tokens)
    bwd = _factor_count(tokens[::-1])
    total_factors = (fwd + bwd) / 2
    return len(tokens) / total_factors if total_factors > 0 else 0.0


def _filler_ratio(text: str, token_count: int) -> float:
    if token_count == 0:
        return 0.0
    count = len(_FILLER_PATTERN.findall(text))
    return count / token_count


def _repetition_ratio(tokens: list[str], window: int = 5) -> float:
    """直近 window トークン内に同じトークンが出る割合。"""
    if len(tokens) < window:
        return 0.0
    hits = 0
    for i in range(window, len(tokens)):
        if tokens[i] in tokens[i - window:i]:
            hits += 1
    return hits / (len(tokens) - window)


def extract_linguistic_features(
    text: str,
    pos_tagged: list[tuple[str, str]],  # [(lemma, pos), ...]
) -> LinguisticFeatures:
    """表層言語特徴量を抽出する。

    Args:
        text: cha_parser 出力の書き起こしテキスト（フィラー検出・文長算出用）。
            文境界（. ? !）を保持していること。text_normalizer.normalize() は
            句読点を除去して文分割を壊すため、ここには通さないこと。
        pos_tagged: src.preprocessing.tokenizer.pos_tag() の出力
    """
    tokens = [lemma for lemma, _ in pos_tagged]
    pos_list = [pos for _, pos in pos_tagged]
    n = len(tokens) or 1

    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    # 英語は語数で文長を測る
    avg_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0.0

    return LinguisticFeatures(
        ttr=_ttr(tokens),
        mtld=_mtld(tokens),
        avg_sentence_len=avg_len,
        noun_ratio=pos_list.count("NOUN") / n,
        verb_ratio=pos_list.count("VERB") / n,
        adj_ratio=(pos_list.count("ADJ") + pos_list.count("ADV")) / n,
        filler_ratio=_filler_ratio(text, n),
        repetition_ratio=_repetition_ratio(tokens),
    )
