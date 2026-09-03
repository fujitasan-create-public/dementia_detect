"""被験者単位10-fold CVでRandomForest/XGBoostのハイパーパラメータ探索を行う。

既存の src.classification.classifier.evaluate() (StratifiedGroupKFold, random_state=42固定)
をそのまま使い、複数パラメータ設定でCV平均AUC/F1を比較する。同一foldなので設定間の比較は公平だが、
CVそのものへの過適合(同じfoldで選び続けるリスク)は残る点に注意。

使い方:
    python scripts/tune_classifier.py --data data/processed/pitt_cookie_features.csv
"""
import sys
import argparse
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.classification.classifier import evaluate, subject_id

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
y = df["label"].values
groups = df["id"].apply(subject_id).values
X = df.drop(columns=["label", "id"], errors="ignore").fillna(0).values

results = []

rf_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 15],
    "max_features": ["sqrt", 0.5, "log2"],
    "min_samples_leaf": [1, 2, 4],
}
for n_estimators, max_depth, max_features, min_samples_leaf in product(
    rf_grid["n_estimators"], rf_grid["max_depth"], rf_grid["max_features"], rf_grid["min_samples_leaf"]
):
    from src.classification import classifier as clf_mod
    clf_mod.MODELS["_tmp"] = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    m = evaluate(X, y, groups, "_tmp")
    m.update(model="random_forest", n_estimators=n_estimators, max_depth=max_depth,
              max_features=max_features, min_samples_leaf=min_samples_leaf)
    results.append(m)

xgb_grid = {
    "n_estimators": [100, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
for n_estimators, max_depth, learning_rate, subsample, colsample_bytree in product(
    xgb_grid["n_estimators"], xgb_grid["max_depth"], xgb_grid["learning_rate"],
    xgb_grid["subsample"], xgb_grid["colsample_bytree"],
):
    from src.classification import classifier as clf_mod
    clf_mod.MODELS["_tmp"] = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        eval_metric="logloss",
    )
    m = evaluate(X, y, groups, "_tmp")
    m.update(model="xgboost", n_estimators=n_estimators, max_depth=max_depth,
              learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree)
    results.append(m)

res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
res_df.to_csv("data/processed/tune_results.csv", index=False)
print(f"Total configs tried: {len(res_df)}")
print("\n=== Top 10 by AUC ===")
print(res_df.head(10).to_string(index=False))
