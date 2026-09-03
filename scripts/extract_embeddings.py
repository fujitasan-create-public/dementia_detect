"""対話CSV(id,text,label)から埋め込みベクトル(nomic-embed-text, 768次元)を抽出して保存する。

使い方:
    python scripts/extract_embeddings.py --data data/raw/pitt_cookie_dataset.csv --out data/processed/pitt_cookie_embeddings.csv
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.features.embedding_features import OllamaEmbedClient, embed_long_text

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--out", type=str, required=True)
args = parser.parse_args()

client = OllamaEmbedClient()

df = pd.read_csv(args.data)
if args.limit:
    df = df.iloc[: args.limit]

rows, ids, labels = [], [], []
t_start = time.time()
for i, row in df.iterrows():
    vec = embed_long_text(row["text"], client)
    rows.append(vec)
    ids.append(row["id"])
    labels.append(row["label"])
    print(f"{len(ids)}/{len(df)} id={row['id']} elapsed={time.time()-t_start:.1f}s dim={len(vec)}", flush=True)

dim = len(rows[0])
out_df = pd.DataFrame(rows, columns=[f"emb_{i}" for i in range(dim)])
out_df.insert(0, "id", ids)
out_df["label"] = labels
out_df.to_csv(args.out, index=False)
print(f"\nSaved {out_df.shape} to {args.out}")
print(f"Total elapsed: {time.time() - t_start:.1f}s")
