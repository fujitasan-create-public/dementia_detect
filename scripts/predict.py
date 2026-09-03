"""新規の対話テキスト(Cookie Theft絵描写課題を想定)から認知機能低下の判定を行う。

予測確率は embedding+SVM（CV/ホールドアウト検証済み、AUC~0.86）から、
判定理由の説明文は LLM26特徴+言語8特徴+RandomForest+SHAP から生成する
（2種類のモデルの役割分担は memory: pipeline-status-2026-09 参照）。

事前に `python scripts/train_final_model.py` でモデルを学習・保存しておくこと。
Ollama（ネイティブ版）が起動しており、qwen2.5:7bとnomic-embed-textがpull済みであること。

使い方:
    python scripts/predict.py --text "the mother is washing dishes..."
    python scripts/predict.py --file path/to/transcript.txt
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.text_normalizer import normalize
from src.preprocessing.tokenizer import pos_tag
from src.features.llm_features import extract_llm_features, OllamaClient
from src.features.linguistic_features import extract_linguistic_features
from src.features.embedding_features import OllamaEmbedClient, embed_long_text
from src.explainability.shap_analyzer import compute_shap_values, generate_explanation

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--text", type=str, help="対話テキスト（Cookie Theft絵描写課題を想定）")
group.add_argument("--file", type=str, help="対話テキストファイルのパス")
args = parser.parse_args()

if args.file:
    raw_text = Path(args.file).read_text(encoding="utf-8")
else:
    raw_text = args.text

svm_pipeline = joblib.load(MODELS_DIR / "pitt_cookie_embedding_svm.joblib")
rf_bundle = joblib.load(MODELS_DIR / "pitt_cookie_features_rf.joblib")
rf_model, feature_columns = rf_bundle["model"], rf_bundle["feature_columns"]

print("Extracting embedding...", flush=True)
emb_client = OllamaEmbedClient()
embedding = embed_long_text(raw_text, emb_client)
if not embedding:
    sys.exit("Embedding extraction failed (Ollama unreachable or empty response).")

print("Extracting LLM + linguistic features (~20s)...", flush=True)
llm_client = OllamaClient(model="qwen2.5:7b")
tagged = pos_tag(normalize(raw_text))
llm_feat = extract_llm_features(raw_text, llm_client.generate)
ling_feat = extract_linguistic_features(raw_text, tagged)
feat_row = {**{f"llm_{k}": v for k, v in llm_feat.to_dict().items()},
            **{f"ling_{k}": v for k, v in ling_feat.to_dict().items()}}
X_feat = pd.DataFrame([feat_row]).reindex(columns=feature_columns, fill_value=0.0).values

X_emb = np.array([embedding])
proba = svm_pipeline.predict_proba(X_emb)[0, 1]
label = "cognitively impaired" if proba >= 0.5 else "cognitively healthy"

shap_values = compute_shap_values(rf_model, X_feat)
explanation = generate_explanation(shap_values[0], feature_columns)

print("\n=== Prediction (embedding + SVM, AUC~0.86 on held-out validation) ===")
print(f"Label: {label}")
print(f"Probability of cognitive decline: {proba:.3f}")
print("\n=== Explanation (interpretable features + SHAP) ===")
print(explanation)
