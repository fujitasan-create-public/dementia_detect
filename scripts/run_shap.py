"""features.csvに対してRandom Forestを学習し、SHAP分析(重要度ランキング・plot・個別説明文)を出力する。

使い方:
    python scripts/run_shap.py --data data/processed/features.csv --out-dir data/processed
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.explainability.shap_analyzer import compute_shap_values, plot_summary, generate_explanation

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="features.csv (id, ..., label列)")
parser.add_argument("--out-dir", type=str, default=".")
args = parser.parse_args()

df = pd.read_csv(args.data)
y = df["label"].values
X = df.drop(columns=["label", "id"], errors="ignore").fillna(0)
feature_names = list(X.columns)

model = RandomForestClassifier(n_estimators=100, max_features=0.5, random_state=42)
model.fit(X, y)

shap_values = compute_shap_values(model, X)
print("shap_values shape:", shap_values.shape)

mean_abs = np.abs(shap_values).mean(axis=0)
ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
print("\n=== Global feature importance (mean |SHAP|), top 15 ===")
print(ranking.head(15).to_string())

plot_summary(shap_values, X, output_path=f"{args.out_dir}/shap_bar.png")

proba = model.predict_proba(X)[:, 1]
pred = model.predict(X)
tp_idx = np.where((y == 1) & (pred == 1))[0]
tn_idx = np.where((y == 0) & (pred == 0))[0]
if len(tp_idx):
    tp_best = tp_idx[np.argmax(proba[tp_idx])]
    print("\n=== Example explanation: confident TRUE POSITIVE (predicted decline) ===")
    print("id:", df.iloc[tp_best]["id"], "proba:", round(proba[tp_best], 3))
    print(generate_explanation(shap_values[tp_best], feature_names))
if len(tn_idx):
    tn_best = tn_idx[np.argmin(proba[tn_idx])]
    print("\n=== Example explanation: confident TRUE NEGATIVE (predicted control) ===")
    print("id:", df.iloc[tn_best]["id"], "proba:", round(proba[tn_best], 3))
    print(generate_explanation(shap_values[tn_best], feature_names))

print("\nSaved plot to:", f"{args.out_dir}/shap_bar.png")
