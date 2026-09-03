"""被験者単位で1回だけtrain/testに分割し、CVで選んだ設定がホールドアウトでも通用するか確認する。

evaluate()による10-fold CV(random_state=42)は設定選び(モデル比較)に使った分割そのものなので、
ここでは別のrandom_state(123)で被験者グループを分割し、独立した検証とする。

使い方:
    python scripts/eval_holdout.py \
        --embeddings data/processed/pitt_cookie_embeddings.csv \
        --features data/processed/pitt_cookie_features.csv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.classification.classifier import subject_id

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", required=True)
parser.add_argument("--features", required=True)
parser.add_argument("--random-state", type=int, default=123, help="CVモデル選定時(42)とは別の値にする")
args = parser.parse_args()

emb_df = pd.read_csv(args.embeddings)
feat_df = pd.read_csv(args.features)
merged = pd.merge(emb_df, feat_df, on=["id", "label"], suffixes=("", "_feat"))

emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
feat_cols = [c for c in feat_df.columns if c not in ("id", "label")]

y = merged["label"].values
groups = merged["id"].apply(subject_id).values

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.random_state)
train_idx, test_idx = next(sgkf.split(merged, y, groups))

assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "被験者リークがあります"
print(f"train: {len(train_idx)}件 ({len(set(groups[train_idx]))}被験者) / test: {len(test_idx)}件 ({len(set(groups[test_idx]))}被験者)")
print(f"train label balance: {np.bincount(y[train_idx])}  test label balance: {np.bincount(y[test_idx])}")

datasets = {
    "embedding": merged[emb_cols].fillna(0).values,
    "combined": merged[emb_cols + feat_cols].fillna(0).values,
    "features_only": merged[feat_cols].fillna(0).values,
}

configs = [
    ("emb_pca50_svm_rbf", "embedding", Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ])),
    ("emb_pca50_logreg", "embedding", Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])),
    ("combined_pca50_logreg", "combined", Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])),
    ("feat_only_rf_tuned_baseline", "features_only", Pipeline([
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=5, max_features="sqrt", min_samples_leaf=4, random_state=42)),
    ])),
]

print("\n=== Holdout test results (train fit -> unseen test predict) ===")
results = []
for name, dataset_key, pipeline in configs:
    X = datasets[dataset_key]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_prob = pipeline.predict_proba(X_te)[:, 1]
    m = {
        "config": name,
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "auc": roc_auc_score(y_te, y_prob),
    }
    results.append(m)
    print(f"{name:32s} f1={m['f1']:.4f}  auc={m['auc']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}")

pd.DataFrame(results).to_csv("data/processed/holdout_eval_results.csv", index=False)
