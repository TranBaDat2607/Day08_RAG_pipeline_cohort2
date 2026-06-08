"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Thiết kế (đồng bộ tuyệt đối với Task 4):
    - Cùng embedding model intfloat/multilingual-e5-small (384 dim) và cùng
      normalize_embeddings=True -> query vector cùng không gian với passage vector.
    - Thêm prefix "query: " cho query (quy ước họ E5; passage đã dùng "passage: "
      khi index). Sai/thiếu prefix sẽ làm giảm rõ rệt chất lượng dense retrieval.
    - Đọc cùng ChromaDB collection "DrugLawDocs" (cosine) mà Task 4 đã ghi.
    - Chroma trả 'distance' cosine in [0, 2]; similarity = 1 - distance -> càng
      lớn càng liên quan, khớp yêu cầu "sorted descending".

Tối ưu: model SentenceTransformer được load 1 lần (lazy, cache module-level) để
các lần gọi sau (và khi Task 9 gọi nhiều lần) không phải nạp lại model ~470MB.
"""

from functools import lru_cache

from src.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBED_QUERY_PREFIX,
    normalize_text,
)


@lru_cache(maxsize=1)
def _get_model():
    """Load embedding model 1 lần và cache (tránh nạp lại ~470MB mỗi query)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    """Mở Chroma collection mà Task 4 đã index và cache client."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (1 - distance)
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    # Bước 0: chuẩn hoá query giống lúc index (NFC + gọn whitespace) cho nhất quán.
    query = normalize_text(query)
    if not query:
        return []

    # Bước 1: Embed query bằng CÙNG model ở Task 4 + prefix "query: " của họ E5.
    model = _get_model()
    query_embedding = model.encode(
        EMBED_QUERY_PREFIX + query,
        normalize_embeddings=True,
    ).tolist()

    # Bước 2: Query vector store (cosine). Lấy đúng top_k kết quả gần nhất.
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma trả batched lists (1 phần tử/query). Collection rỗng -> trả [].
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    # Bước 3: distance -> similarity, đóng gói đúng contract dict.
    output = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        output.append({
            "content": content,
            "score": 1.0 - distance,          # cosine distance -> similarity
            "metadata": dict(metadata or {}),
        })

    # Chroma đã trả theo distance tăng dần (gần nhất trước) = score giảm dần,
    # nhưng sort lại lần nữa để đảm bảo contract "sorted descending".
    output.sort(key=lambda r: r["score"], reverse=True)
    return output


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
