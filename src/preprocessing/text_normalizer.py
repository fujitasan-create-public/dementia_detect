"""英語テキスト正規化（Unicode正規化、小文字化、記号除去）。

CHAT の書き起こしは cha_parser 側でマークアップ除去済み。ここでは特徴抽出
のために表記を揃える（大文字小文字・スマートクォート・余分な記号の吸収）。
"""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # スマートクォート等を単純アポストロフィに寄せる
    text = text.replace("’", "'").replace("‘", "'")
    # 英数字・空白・語中のアポストロフィ/ハイフン以外を空白化
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    # 単独で浮いたアポストロフィ/ハイフンを除去
    text = re.sub(r"(?<!\w)['\-]+|['\-]+(?!\w)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
