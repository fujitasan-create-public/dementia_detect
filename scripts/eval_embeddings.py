"""埋め込み(nomic-embed-text 768次元)を使った分類性能を、既存のLLM+言語特徴と比較する。

漏洩を避けるため、PCA/StandardScalerは全てPipelineに組み込み、
被験者単位10-fold CV(既存のevaluate())の各foldでtrainのみに対してfitする。

使い方:
    python scripts/eval_embeddings.py \
        --embeddings data/processed/pitt_cookie_embeddings.csv \
        --features data/processed/pitt_cookie_features.csv
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.classification import classifier as clf_mod
from src.classification.classifier import evaluate, subject_id

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", required=True)
parser.add_argument("--features", required=True)
parser.add_argument("--out", default="data/processed/embedding_eval_results.csv")
args = parser.parse_args()

emb_df = pd.read_csv(args.embeddings)
feat_df = pd.read_csv(args.features)

emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
merged = pd.merge(emb_df, feat_df, on=["id", "label"], suffixes=("", "_feat"))
feat_cols = [c for c in feat_df.columns if c not in ("id", "label")]

y = merged["label"].values
groups = merged["id"].apply(subject_id).values

X_emb = merged[emb_cols].fillna(0).values
X_feat = merged[feat_cols].fillna(0).values
X_combined = merged[emb_cols + feat_cols].fillna(0).values

configs = [
    ("emb_pca50_logreg", X_emb, Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])),
    ("emb_pca100_logreg", X_emb, Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=100, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])),
    ("emb_raw_logreg_l2", X_emb, Pipeline([
        ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, C=0.1)),
    ])),
    ("emb_pca50_svm_rbf", X_emb, Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
    ])),
    ("emb_pca50_rf", X_emb, Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)),
    ])),
    ("combined_pca50_logreg", X_combined, Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])),
    ("combined_raw_rf_tuned", X_combined, Pipeline([
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=5, max_features="sqrt", min_samples_leaf=4, random_state=42)),
    ])),
    ("combined_raw_xgb_tuned", X_combined, Pipeline([
        ("clf", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0, random_state=42, eval_metric="logloss")),
    ])),
    ("feat_only_rf_tuned_baseline", X_feat, Pipeline([
        ("clf", RandomForestClassifier(n_estimators=500, max_depth=5, max_features="sqrt", min_samples_leaf=4, random_state=42)),
    ])),
]

results = []
for name, X, pipeline in configs:
    clf_mod.MODELS[name] = pipeline
    m = evaluate(X, y, groups, name)
    m["config"] = name
    m["n_features"] = X.shape[1]
    results.append(m)
    print(f"{name:32s} n_feat={X.shape[1]:4d}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  precision={m['precision']:.4f}  recall={m['recall']:.4f}")

res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
res_df.to_csv(args.out, index=False)
print("\n=== Sorted by AUC ===")
print(res_df[["config", "n_features", "f1", "auc", "precision", "recall"]].to_string(index=False))
