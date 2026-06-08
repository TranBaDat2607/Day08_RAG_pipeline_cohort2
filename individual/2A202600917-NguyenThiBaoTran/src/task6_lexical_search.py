"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

from .task4_chunking_indexing import chunk_documents, load_documents

# Corpus dùng chung 1 cách chunk với Task 4 (load_documents + chunk_documents)
# để lexical search và semantic search trả về cùng đơn vị "chunk" — cần thiết
# nếu sau này kết hợp 2 kết quả thành hybrid search (so điểm/merge theo chunk).
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

# BM25 index được build 1 lần và tái sử dụng — tránh tokenize + tính lại toàn
# bộ corpus cho mỗi query.
_bm25_index: BM25Okapi | None = None


def _load_corpus() -> list[dict]:
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def _tokenize(text: str) -> list[str]:
    """
    Tokenize tiếng Việt bằng underthesea.word_tokenize thay vì split() đơn
    thuần: tiếng Việt là ngôn ngữ đơn lập, nhiều từ ghép gồm 2+ âm tiết cách
    nhau bởi khoảng trắng (vd. "ma túy", "tàng trữ", "trái phép"). split() sẽ
    tách rời các âm tiết này thành token riêng lẻ, làm BM25 đánh giá sai
    TF/IDF (so khớp từng âm tiết chung chung thay vì khái niệm trọn vẹn).
    underthesea.word_tokenize gộp các âm tiết thành 1 token từ ghép duy nhất
    (vd. "tàng trữ", "trái phép" → mỗi cụm là 1 phần tử trong list trả về),
    giúp BM25 khớp đúng thuật ngữ pháp lý/chuyên ngành trọn vẹn thay vì so
    từng âm tiết rời rạc.
    """
    tokens = word_tokenize(text.lower())
    # Loại token chỉ gồm dấu câu/khoảng trắng (không chứa ký tự chữ/số)
    return [t for t in tokens if re.search(r"\w", t, flags=re.UNICODE)]


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _bm25_index

    corpus = _load_corpus()
    if _bm25_index is None:
        _bm25_index = build_bm25_index(corpus)

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    import numpy as np

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
