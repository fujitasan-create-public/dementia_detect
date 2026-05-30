"""LLM特徴量と言語特徴量を結合し、Pearson相関によって特徴選択を行う。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def merge_features(
    llm_dicts: list[dict[str, float]],
    ling_dicts: list[dict[str, float]],
) -> pd.DataFrame:
    """LLM特徴と言語特徴を横結合した DataFrame を返す。"""
    llm_df = pd.DataFrame(llm_dicts).add_prefix("llm_")
    ling_df = pd.DataFrame(ling_dicts).add_prefix("ling_")
    return pd.concat([llm_df, ling_df], axis=1)


def select_by_pearson(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.1,
) -> list[str]:
    """ラベルとのPearson相関が threshold 以上の特徴名リストを返す。"""
    selected = []
    for col in X.columns:
        corr, _ = pearsonr(X[col].fillna(0), y)
        if abs(corr) >= threshold:
            selected.append(col)
    return selected
