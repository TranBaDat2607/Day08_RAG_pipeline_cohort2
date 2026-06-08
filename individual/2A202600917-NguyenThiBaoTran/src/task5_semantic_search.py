"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from sentence_transformers import SentenceTransformer
from weaviate.classes.query import MetadataQuery

from .weaviate_client import connect_weaviate

# Phải khớp với cấu hình ở Task 4 (cùng model để query vector nằm chung không
# gian embedding với dữ liệu đã index, cùng tên collection để truy vấn đúng nơi).
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "DrugLawDocs"

# Load model một lần và tái sử dụng giữa các lần gọi semantic_search — tránh
# tốn thời gian/bộ nhớ load lại model cho mỗi query.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    model = _get_model()
    # normalize_embeddings=True để khớp với cách embed chunk ở Task 4 — vector
    # đã chuẩn hóa nên distance trả về của Weaviate (cosine distance) chuyển
    # thẳng sang similarity bằng "1 - distance".
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    client = connect_weaviate()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        return [
            {
                "content": obj.properties["content"],
                "score": 1 - obj.metadata.distance,
                "metadata": {
                    "source": obj.properties["source"],
                    "doc_type": obj.properties["doc_type"],
                    "chunk_index": obj.properties["chunk_index"],
                },
            }
            for obj in results.objects
        ]
    finally:
        client.close()


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
