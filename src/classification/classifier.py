"""RandomForest / DecisionTree / NaiveBayes による 10-fold CV 分類。"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "naive_bayes": GaussianNB(),
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "xgboost": XGBClassifier(random_state=42, eval_metric="logloss"),
    "lightgbm": LGBMClassifier(random_state=42, verbose=-1),
}


def subject_id(record_id: str) -> str:
    """同一被験者の複数セッションIDを束ねる（例: '01-2' -> '01'）。

    DementiaBank の一部ソース（Delaware 等）は同一被験者から複数セッション分の
    書き起こしを持つため、これでグルーピングしないと交差検証で同一被験者が
    train/val 両方に混入しリークする。
    """
    return re.sub(r"-\d+$", "", str(record_id))


def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str = "random_forest",
) -> dict:
    """被験者単位でグループ化した 10-fold 層化交差検証でメトリクスを返す。"""
    model = MODELS[model_name]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"f1": [], "precision": [], "recall": [], "auc": []}

    for train_idx, val_idx in sgkf.split(X, y, groups):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics["f1"].append(f1_score(y_val, y_pred, zero_division=0))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred, zero_division=0))
        metrics["auc"].append(roc_auc_score(y_val, y_prob))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="特徴量CSV (id, ..., label 列を含む)")
    parser.add_argument("--model", default="random_forest", choices=list(MODELS))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df["label"].values
    groups = df["id"].apply(subject_id).values
    X = df.drop(columns=["label", "id"], errors="ignore").fillna(0).values

    result = evaluate(X, y, groups, args.model)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
