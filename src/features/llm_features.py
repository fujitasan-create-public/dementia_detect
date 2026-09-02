"""ローカルLLMを用いた高次推論特徴の抽出（26特徴、0.0–1.0スコア）。

使用モデル: Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct 等（Ollama で切り替え）
実行基盤: Ollama（ローカル HTTP サーバ, 既定 http://localhost:11434）
対象言語: 英語（DementiaBank 書き起こし）

呼び出し側は `OllamaClient` を生成し、その `.generate` を渡す:

    client = OllamaClient(model="qwen2.5:7b")
    feat = extract_llm_features(dialogue_text, client.generate)
"""
from __future__ import annotations
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, fields
from typing import Callable

# 特徴名は de Arriba-Pérez ら (2024) の 26 特徴（英語対話向け）
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
You are an expert in cognitive function and language assessment.
Read the following dialogue (investigator and participant utterances) and
rate each of the specified features on a scale from 0.0 to 1.0.
Output ONLY a JSON object with the scores. Do not add any explanation.

[Feature list]
{feature_list}

[Dialogue]
{dialogue}

Output format: {{"feature_name": score, ...}}
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


class OllamaClient:
    """Ollama のローカル HTTP サーバに繋いでテキスト生成する薄いクライアント。

    標準ライブラリ（urllib）だけで実装。事前に別ターミナルで `ollama serve`
    が起動し、対象モデルが pull 済みであること（例: `ollama pull qwen2.5:7b`）。
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int = 256,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """プロンプトを投げて生成テキストを返す。失敗時は空文字。

        `format="json"` で JSON 出力を強制し、後段のパース失敗を減らす。
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""


def extract_llm_features(
    dialogue_text: str,
    generate: Callable[[str], str],
) -> LLMFeatures:
    """対話テキストからLLM特徴を抽出する。

    Args:
        dialogue_text: 前処理済みの対話テキスト
        generate: プロンプト(str)を受けて生成テキスト(str)を返す関数。
            通常は `OllamaClient(...).generate` を渡す。
    """
    prompt = _build_prompt(dialogue_text)
    generated = generate(prompt)
    scores = _parse_scores(generated)
    feat = LLMFeatures()
    for name, val in scores.items():
        if hasattr(feat, name):
            setattr(feat, name, max(0.0, min(1.0, val)))
    return feat
