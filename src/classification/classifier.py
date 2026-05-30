"""RandomForest / DecisionTree / NaiveBayes による 10-fold CV 分類。"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


MODELS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "naive_bayes": GaussianNB(),
}


def evaluate(X: np.ndarray, y: np.ndarray, model_name: str = "random_forest") -> dict:
    """10-fold 層化交差検証でメトリクスを返す。"""
    model = MODELS[model_name]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    metrics = {"f1": [], "precision": [], "recall": [], "auc": []}

    for train_idx, val_idx in skf.split(X, y):
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
    parser.add_argument("--data", required=True, help="特徴量CSV (最終列が label)")
    parser.add_argument("--model", default="random_forest", choices=list(MODELS))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    y = df["label"].values
    X = df.drop(columns=["label"]).fillna(0).values

    result = evaluate(X, y, args.model)
    for k, v in result.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
