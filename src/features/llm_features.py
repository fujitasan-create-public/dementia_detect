"""ローカルLLMを用いた高次推論特徴の抽出（26特徴、0.0–1.0スコア）。

使用モデル: Qwen2.5-7B-Instruct / Swallow-7B-Instruct 等（config で切り替え）
量子化: bitsandbytes 4bit NF4
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, fields

# 特徴名は de Arriba-Pérez ら (2024) の 26 特徴を日本語対話向けに調整
FEATURE_NAMES: list[str] = [
    "memory_impairment",
    "topic_drift",
    "initiative_lack",
    "mood_change",
    "happiness_expression",
    "comprehension_difficulty",
    "expression_difficulty",
    "repetitive_language",
    "short_responses",
    "complex_vocabulary",
    "temporal_confusion",
    "person_confusion",
    "word_finding_difficulty",
    "tangential_speech",
    "filler_frequency",
    "self_correction",
    "perseveration",
    "reduced_detail",
    "semantic_paraphasia",
    "phonemic_paraphasia",
    "circumlocution",
    "echo_response",
    "inappropriate_affect",
    "social_withdrawal",
    "fatigue_signs",
    "confabulation",
]


@dataclass
class LLMFeatures:
    memory_impairment: float = 0.0
    topic_drift: float = 0.0
    initiative_lack: float = 0.0
    mood_change: float = 0.0
    happiness_expression: float = 0.0
    comprehension_difficulty: float = 0.0
    expression_difficulty: float = 0.0
    repetitive_language: float = 0.0
    short_responses: float = 0.0
    complex_vocabulary: float = 0.0
    temporal_confusion: float = 0.0
    person_confusion: float = 0.0
    word_finding_difficulty: float = 0.0
    tangential_speech: float = 0.0
    filler_frequency: float = 0.0
    self_correction: float = 0.0
    perseveration: float = 0.0
    reduced_detail: float = 0.0
    semantic_paraphasia: float = 0.0
    phonemic_paraphasia: float = 0.0
    circumlocution: float = 0.0
    echo_response: float = 0.0
    inappropriate_affect: float = 0.0
    social_withdrawal: float = 0.0
    fatigue_signs: float = 0.0
    confabulation: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


_PROMPT_TEMPLATE = """\
あなたは認知機能と言語の専門家です。
以下の対話ペア（ボット発話・ユーザ発話）を読み、
指定された特徴について 0.0〜1.0 のスコアを返してください。
スコアは JSON フォーマットのみで出力し、説明は不要です。

[特徴リスト]
{feature_list}

[対話]
{dialogue}

出力形式: {{"feature_name": score, ...}}
"""


def _build_prompt(dialogue_text: str) -> str:
    feature_list = "\n".join(f"- {n}" for n in FEATURE_NAMES)
    return _PROMPT_TEMPLATE.format(
        feature_list=feature_list,
        dialogue=dialogue_text,
    )


def _parse_scores(raw: str) -> dict[str, float]:
    """LLM 出力から JSON を抽出してスコア辞書を返す。"""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group())
        return {k: float(v) for k, v in data.items() if k in FEATURE_NAMES}
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_llm_features(
    dialogue_text: str,
    pipeline,  # transformers.pipeline("text-generation", ...)
) -> LLMFeatures:
    """対話テキストからLLM特徴を抽出する。

    Args:
        dialogue_text: 前処理済みの対話テキスト
        pipeline: transformers の text-generation パイプライン（呼び出し元で初期化）
    """
    prompt = _build_prompt(dialogue_text)
    output = pipeline(prompt, max_new_tokens=256, do_sample=False)
    generated = output[0]["generated_text"][len(prompt):]
    scores = _parse_scores(generated)
    feat = LLMFeatures()
    for name, val in scores.items():
        if hasattr(feat, name):
            setattr(feat, name, max(0.0, min(1.0, val)))
    return feat
