"""埋め込みベースの分類パイプライン(PCA次元数, SVM/LogRegハイパーパラメータ)を
被験者単位10-fold CVでグリッドサーチする。

使い方:
    python scripts/tune_embeddings.py --embeddings data/processed/pitt_cookie_embeddings.csv
"""
import sys
import argparse
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.classification import classifier as clf_mod
from src.classification.classifier import evaluate, subject_id

parser = argparse.ArgumentParser()
parser.add_argument("--embeddings", required=True)
parser.add_argument("--out", default="data/processed/tune_embeddings_results.csv")
args = parser.parse_args()

df = pd.read_csv(args.embeddings)
emb_cols = [c for c in df.columns if c.startswith("emb_")]
y = df["label"].values
groups = df["id"].apply(subject_id).values
X = df[emb_cols].fillna(0).values

results = []

pca_options = [30, 50, 75, 100]
svm_C = [0.1, 1, 10, 100]
svm_gamma = ["scale", 0.001, 0.01, 0.1]
logreg_C = [0.01, 0.1, 1, 10, 100]

for n_pca, C, gamma in product(pca_options, svm_C, svm_gamma):
    pipe = Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=n_pca, random_state=42)),
        ("clf", SVC(C=C, gamma=gamma, kernel="rbf", probability=True, random_state=42)),
    ])
    clf_mod.MODELS["_tmp"] = pipe
    m = evaluate(X, y, groups, "_tmp")
    m.update(model="svm_rbf", n_pca=n_pca, C=C, gamma=gamma)
    results.append(m)

for n_pca, C in product(pca_options, logreg_C):
    pipe = Pipeline([
        ("scaler", StandardScaler()), ("pca", PCA(n_components=n_pca, random_state=42)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])
    clf_mod.MODELS["_tmp"] = pipe
    m = evaluate(X, y, groups, "_tmp")
    m.update(model="logreg", n_pca=n_pca, C=C, gamma=None)
    results.append(m)

res_df = pd.DataFrame(results).sort_values("auc", ascending=False)
res_df.to_csv(args.out, index=False)
print(f"Total configs tried: {len(res_df)}")
print("\n=== Top 15 by AUC ===")
print(res_df.head(15).to_string(index=False))
