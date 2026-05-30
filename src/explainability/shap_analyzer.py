"""SHAP による特徴重要度の可視化と自然言語テンプレート出力。"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_shap_values(model, X: np.ndarray):
    """TreeExplainer で SHAP 値を計算して返す。"""
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def plot_summary(shap_values, X: pd.DataFrame, output_path: str | None = None):
    """SHAP summary plot を表示 or ファイル保存する。"""
    import shap
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, X, show=output_path is None)
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()


_FEATURE_LABEL_JA = {
    "llm_memory_impairment": "記憶障害",
    "llm_repetitive_language": "繰り返し発話",
    "llm_topic_drift": "話題逸脱",
    "ling_ttr": "語彙多様性(TTR)",
    "ling_filler_ratio": "フィラー頻度",
}


def generate_explanation(shap_row: np.ndarray, feature_names: list[str], top_n: int = 3) -> str:
    """1件分のSHAP値から自然言語の判定理由文を生成する。"""
    idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    reasons = []
    for i in idx:
        name = feature_names[i]
        label = _FEATURE_LABEL_JA.get(name, name)
        direction = "高い" if shap_row[i] > 0 else "低い"
        reasons.append(f"{label}が{direction}")
    return "、".join(reasons) + "ため、認知機能低下の可能性があります。"
