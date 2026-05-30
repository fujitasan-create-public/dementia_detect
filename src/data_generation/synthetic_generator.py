"""ローカルLLMを使った認知症/正常対話の合成データ生成。

LLMCARE (Zolnour et al., 2025) の Component 2 を日本語向けに再実装。
"""
from __future__ import annotations
import json
import random
from pathlib import Path


_PROMPT_TEMPLATE = """\
あなたは言語と認知の専門家です。
高齢者の介護施設でチャットボットと交わされる会話を生成してください。
以下の特性を持つユーザの発話を含む対話を作成してください。

[ラベル] {label_desc}
[特性]
{characteristics}

出力形式: {{"bot": "...", "user": "..."}} の JSON を{turns}ターン分
"""

_CHARACTERISTICS = {
    1: {
        "label_desc": "認知機能低下あり",
        "chars": [
            "同じ内容の語句・話題を繰り返す",
            "話の途中で話題が突然変わる",
            "「えーと」「あのー」などフィラーが多い",
            "過去の出来事と現在を混同する",
            "短く単調な返答が多い",
        ],
    },
    0: {
        "label_desc": "認知機能正常",
        "chars": [
            "話題が一貫している",
            "複雑な語彙を適切に使用する",
            "過去の出来事を正確に語れる",
            "自発的に話題を展開する",
        ],
    },
}


def build_prompt(label: int, turns: int = 5) -> str:
    info = _CHARACTERISTICS[label]
    chars = "\n".join(f"- {c}" for c in info["chars"])
    return _PROMPT_TEMPLATE.format(
        label_desc=info["label_desc"],
        characteristics=chars,
        turns=turns,
    )


def generate_synthetic_dialogue(
    label: int,
    pipeline,
    turns: int = 5,
) -> dict:
    """合成対話を1件生成して辞書で返す。

    Returns:
        {"label": int, "turns": [{"bot": str, "user": str}, ...]}
    """
    prompt = build_prompt(label, turns)
    output = pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.8)
    generated = output[0]["generated_text"][len(prompt):]

    # JSON 配列を探して解析
    import re
    match = re.search(r"\[.*\]", generated, re.DOTALL)
    if match:
        try:
            turns_data = json.loads(match.group())
            return {"label": label, "turns": turns_data}
        except json.JSONDecodeError:
            pass
    return {"label": label, "turns": [], "raw": generated}


def save_to_jsonl(records: list[dict], output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
