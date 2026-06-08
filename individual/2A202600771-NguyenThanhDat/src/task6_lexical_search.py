"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25
"""

import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.task4_chunking_indexing import get_index

_bm25: BM25Okapi | None = None
_corpus: list[dict] = []


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _ensure_bm25():
    global _bm25, _corpus
    if _bm25 is not None:
        return

    index = get_index()
    _corpus = index.get("chunks", [])
    if not _corpus:
        return

    tokenized = [_tokenize(c["content"]) for c in _corpus]
    _bm25 = BM25Okapi(tokenized)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _bm25, _corpus
    _corpus = corpus
    if not corpus:
        _bm25 = None
        return
    tokenized = [_tokenize(doc["content"]) for doc in corpus]
    _bm25 = BM25Okapi(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    _ensure_bm25()
    if not _corpus or _bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25.get_scores(tokenized_query)

    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": _corpus[idx].get("metadata", {}),
            })
    return results


# Alias cho code mẫu trong README
CORPUS: list[dict] = []


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
