"""SHAP による特徴重要度の可視化と自然言語テンプレート出力。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_shap_values(model, X: np.ndarray):
    """TreeExplainer で SHAP 値を計算して返す（2値分類の positive class 分に正規化）。

    shap のバージョンによって二値分類の戻り値の形が異なる
    （旧: [class0, class1] のリスト、新: shape (n, features, n_classes) の配列）ため、
    ここで shape (n_samples, n_features) に統一する。
    """
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return sv[1] if len(sv) > 1 else sv[0]
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def plot_summary(shap_values, X: pd.DataFrame, output_path: str | None = None):
    """SHAP summary plot を表示 or ファイル保存する。"""
    import shap
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, show=output_path is None)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


_FEATURE_LABEL = {
    "llm_memory_impairment": "memory impairment",
    "llm_topic_drift": "topic drift",
    "llm_initiative_lack": "lack of initiative",
    "llm_mood_change": "mood change",
    "llm_happiness_expression": "expressed happiness",
    "llm_comprehension_difficulty": "comprehension difficulty",
    "llm_expression_difficulty": "expression difficulty",
    "llm_repetitive_language": "repetitive language",
    "llm_short_responses": "short responses",
    "llm_complex_vocabulary": "complex vocabulary use",
    "llm_temporal_confusion": "temporal confusion",
    "llm_person_confusion": "person confusion",
    "llm_word_finding_difficulty": "word-finding difficulty",
    "llm_tangential_speech": "tangential speech",
    "llm_filler_frequency": "filler frequency",
    "llm_self_correction": "self-correction",
    "llm_perseveration": "perseveration",
    "llm_reduced_detail": "reduced detail",
    "llm_semantic_paraphasia": "semantic paraphasia",
    "llm_phonemic_paraphasia": "phonemic paraphasia",
    "llm_circumlocution": "circumlocution",
    "llm_echo_response": "echoed responses",
    "llm_inappropriate_affect": "inappropriate affect",
    "llm_social_withdrawal": "social withdrawal",
    "llm_fatigue_signs": "signs of fatigue",
    "llm_confabulation": "confabulation",
    "ling_ttr": "lexical diversity (TTR)",
    "ling_mtld": "lexical diversity (MTLD)",
    "ling_avg_sentence_len": "average sentence length",
    "ling_noun_ratio": "noun ratio",
    "ling_verb_ratio": "verb ratio",
    "ling_adj_ratio": "adjective/adverb ratio",
    "ling_filler_ratio": "filler-word ratio",
    "ling_repetition_ratio": "repetition ratio",
}


def generate_explanation(shap_row: np.ndarray, feature_names: list[str], top_n: int = 3) -> str:
    """1件分のSHAP値から自然言語の判定理由文を生成する（英語）。

    結論の方向（低下寄り/正常寄り）は shap_row の合計（class=1 方向への
    正味の寄与）から決める。上位特徴の符号だけを見て常に「低下」と
    結論づけていた旧実装のバグを修正済み。
    """
    idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    reasons = []
    for i in idx:
        name = feature_names[i]
        label = _FEATURE_LABEL.get(name, name)
        direction = "high" if shap_row[i] > 0 else "low"
        reasons.append(f"{direction} {label}")
    conclusion = (
        "suggest a possible indication of cognitive decline"
        if shap_row.sum() > 0
        else "suggest patterns consistent with normal cognitive function"
    )
    return ", ".join(reasons) + f" {conclusion}."
