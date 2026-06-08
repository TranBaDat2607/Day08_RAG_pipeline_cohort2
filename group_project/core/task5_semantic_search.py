"""
Task 5 — Semantic Search Module (group project, backend = FAISS).

Tìm kiếm ngữ nghĩa (dense retrieval) trên vector store FAISS chung của nhóm
(data/faiss/) — bản merge từ bài cá nhân, đổi backend ChromaDB -> FAISS.

Thiết kế (đồng bộ tuyệt đối với Task 4 ingestion của nhóm):
    - Cùng embedding model intfloat/multilingual-e5-small (384d) và cùng
      normalize_embeddings=True -> query vector cùng không gian với passage vector.
    - Thêm prefix "query: " cho query (quy ước họ E5; passage đã dùng "passage: "
      khi index). Sai/thiếu prefix sẽ làm giảm rõ rệt chất lượng dense retrieval.
    - Đọc CÙNG FAISS index (IndexFlatIP + L2-normalize = cosine) mà ingestion đã ghi
      qua get_faiss_store(): index FAISS + danh sách chunks (content + metadata).
    - FAISS trả inner product của vector đã L2-normalize = cosine similarity in
      [-1, 1] -> càng lớn càng liên quan, khớp yêu cầu "sorted descending".

Tối ưu: model SentenceTransformer load 1 lần (lazy, lru_cache) để các lần gọi sau
(và khi Task 9 / Task 7 gọi nhiều lần) không phải nạp lại model ~470MB.
"""

from functools import lru_cache

import numpy as np

from ..ingestion.chunk_and_index import get_faiss_store, normalize_text
from ..ingestion.config import EMBED_QUERY_PREFIX, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model():
    """Load embedding model 1 lần và cache (tránh nạp lại ~470MB mỗi query)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity trên FAISS.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity (inner product sau L2-normalize)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending, độ dài ≤ top_k.
    """
    # Bước 0: chuẩn hoá query giống lúc index (NFC + gọn whitespace) cho nhất quán.
    query = normalize_text(query)
    if not query:
        return []

    # Bước 1: nạp FAISS store (index + chunks) — cache ở tầng ingestion.
    store = get_faiss_store()
    index = store.get("index")
    chunks = store.get("chunks", [])
    if index is None or not chunks or index.ntotal == 0:
        return []

    import faiss

    # Bước 2: Embed query bằng CÙNG model ở Task 4 + prefix "query: " của họ E5.
    query_prefix = store.get("embed_query_prefix", EMBED_QUERY_PREFIX)
    model = _get_model()
    query_vec = model.encode(
        query_prefix + query,
        normalize_embeddings=True,
    ).astype(np.float32)
    # L2-normalize lần nữa để chắc chắn cosine = inner product trên IndexFlatIP.
    faiss.normalize_L2(query_vec.reshape(1, -1))

    # Bước 3: Query FAISS, lấy đúng top_k vector gần nhất.
    scores, indices = index.search(query_vec.reshape(1, -1), min(top_k, index.ntotal))

    # Bước 4: map index -> chunk, đóng gói đúng contract dict.
    output = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        output.append({
            "content": chunk["content"],
            "score": float(score),            # inner product = cosine similarity
            "metadata": dict(chunk.get("metadata", {}) or {}),
        })

    # FAISS đã trả theo score giảm dần, nhưng sort lại để đảm bảo contract.
    output.sort(key=lambda r: r["score"], reverse=True)
    return output


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
