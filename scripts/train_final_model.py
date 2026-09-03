"""Pitt cookie全件で最終モデルを学習し保存する（predict.pyから利用）。

2種類のモデルを保存する:
  - embedding_svm: nomic-embed-text埋め込み(768次元)->PCA50->SVM(RBF)。予測確率の主力（AUC~0.86）。
  - features_rf: LLM26特徴+言語8特徴->RandomForest(チューニング済み設定)。SHAPによる説明用。

使い方:
    python scripts/train_final_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

emb_df = pd.read_csv("data/processed/pitt_cookie_embeddings.csv")
feat_df = pd.read_csv("data/processed/pitt_cookie_features.csv")

emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
feat_cols = [c for c in feat_df.columns if c not in ("id", "label")]

y_emb = emb_df["label"].values
X_emb = emb_df[emb_cols].fillna(0).values

y_feat = feat_df["label"].values
X_feat = feat_df[feat_cols].fillna(0).values

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=50, random_state=42)),
    ("clf", SVC(kernel="rbf", probability=True, random_state=42)),
])
svm_pipeline.fit(X_emb, y_emb)
joblib.dump(svm_pipeline, MODELS_DIR / "pitt_cookie_embedding_svm.joblib")
print(f"Saved embedding SVM pipeline (trained on {len(y_emb)} samples)")

rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=5, max_features="sqrt", min_samples_leaf=4, random_state=42,
)
rf_model.fit(X_feat, y_feat)
joblib.dump({"model": rf_model, "feature_columns": feat_cols}, MODELS_DIR / "pitt_cookie_features_rf.joblib")
print(f"Saved features RF model + explainer bundle (trained on {len(y_feat)} samples, {len(feat_cols)} features)")
