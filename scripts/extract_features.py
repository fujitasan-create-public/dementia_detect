"""対話CSV(id,text,label)からLLM特徴+言語特徴を抽出し、features.csv形式で保存する。

Ollama(Windowsネイティブ版、GPU使用)が起動している前提。1件あたり約20秒。

使い方:
    python scripts/extract_features.py --data data/raw/dataset.csv --out data/processed/features.csv
    python scripts/extract_features.py --data data/raw/pitt_cookie_dataset.csv --out data/processed/pitt_cookie_features.csv --limit 20
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.preprocessing.text_normalizer import normalize
from src.preprocessing.tokenizer import pos_tag
from src.features.llm_features import extract_llm_features, OllamaClient
from src.features.linguistic_features import extract_linguistic_features
from src.features.feature_merger import merge_features

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="入力CSV (id,text,label列)")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

client = OllamaClient(model="qwen2.5:7b")

df = pd.read_csv(args.data)
if args.limit:
    df = df.iloc[: args.limit]

llm_rows, ling_rows, labels, ids = [], [], [], []
t_start = time.time()
for i, row in df.iterrows():
    raw_text = row["text"]
    tagged = pos_tag(normalize(raw_text))
    llm_feat = extract_llm_features(raw_text, client.generate)
    ling_feat = extract_linguistic_features(raw_text, tagged)
    llm_rows.append(llm_feat.to_dict())
    ling_rows.append(ling_feat.to_dict())
    labels.append(row["label"])
    ids.append(row["id"])
    elapsed = time.time() - t_start
    nonzero = sum(1 for v in llm_feat.to_dict().values() if v != 0.0)
    print(f"{len(ids)}/{len(df)} id={row['id']} elapsed={elapsed:.1f}s nonzero_llm={nonzero}/26", flush=True)

out_df = merge_features(llm_rows, ling_rows)
out_df.insert(0, "id", ids)
out_df["label"] = labels
out_df.to_csv(args.out, index=False)
print(f"\nSaved {out_df.shape} to {args.out}")
print(f"Total elapsed: {time.time() - t_start:.1f}s")
