"""Ollamaのembedding APIを使ったテキスト埋め込み抽出。

使用モデル: nomic-embed-text（768次元、英語）
実行基盤: Ollama（ローカル HTTP サーバ, 既定 http://localhost:11434）

    client = OllamaEmbedClient(model="nomic-embed-text")
    vec = client.embed(dialogue_text)  # list[float], 長さ768
"""
from __future__ import annotations
import json
import urllib.error
import urllib.request


class OllamaEmbedClient:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換する。失敗時は空リスト。

        nomic-embed-textの既定コンテキスト長は2048トークンだが、長い書き起こし
        （Delaware等は平均約1500トークン、最大約4300トークン）を切り詰めずに
        扱うため num_ctx=8192 を明示指定する。
        """
        payload = {"model": self.model, "input": text, "options": {"num_ctx": self.num_ctx}}
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            embeddings = body.get("embeddings", [])
            return embeddings[0] if embeddings else []
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return []


def embed_long_text(text: str, client: OllamaEmbedClient, max_words: int = 1400) -> list[float]:
    """モデルの実効コンテキスト長（nomic-embed-textは約2048トークン=1500語強、
    optionsのnum_ctxで拡張不可）を超える長文は語数でチャンク分割し、
    各チャンクの埋め込みを平均・L2正規化して1本のベクトルにする。
    """
    words = text.split()
    if len(words) <= max_words:
        return client.embed(text)

    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    vecs = [v for v in (client.embed(c) for c in chunks) if v]
    if not vecs:
        return []

    dim = len(vecs[0])
    mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
    norm = sum(x * x for x in mean) ** 0.5
    return [x / norm for x in mean] if norm > 0 else mean
