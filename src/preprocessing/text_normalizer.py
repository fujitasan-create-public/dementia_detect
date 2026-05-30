"""日本語テキスト正規化（絵文字・記号除去、Unicode正規化）"""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s぀-ゟ゠-ヿ一-鿿㐀-䶿]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
