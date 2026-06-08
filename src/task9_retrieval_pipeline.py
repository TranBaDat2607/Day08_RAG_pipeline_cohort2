"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: chạy semantic + lexical song song. Cả 2 đều là tác vụ blocking
    # độc lập (encode query + query Weaviate vs tokenize + tính BM25 trên
    # corpus) — chạy đồng thời bằng thread pool giúp giảm tổng độ trễ thay vì
    # chờ tuần tự (I/O- và phần lớn tính toán bên dưới đều giải phóng GIL).
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, top_k * 2)
        sparse_future = executor.submit(lexical_search, query, top_k * 2)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Step 2: merge bằng RRF — 2 ranker có thang điểm KHÔNG so sánh được trực
    # tiếp (cosine similarity ∈ [0,1] vs BM25 score không chặn trên), RRF chỉ
    # dựa vào thứ hạng nên gộp được mà không cần chuẩn hoá điểm (cơ chế chi
    # tiết xem rerank_rrf ở Task 7).
    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    # Step 3: rerank để tăng độ chính xác của top-k cuối — cross-encoder đọc
    # trực tiếp (query, content) nên xếp hạng chuẩn hơn nhiều so với điểm RRF
    # (vốn chỉ phản ánh thứ hạng thô từ 2 ranker, chưa "hiểu" nội dung).
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item.setdefault("source", "hybrid")
    else:
        final_results = merged[:top_k]

    # Step 4: nếu kết quả hybrid không đủ tin cậy (rỗng, hoặc điểm cao nhất
    # dưới ngưỡng) → fallback sang PageIndex. Đây là các trường hợp semantic +
    # lexical đều "đuối" — thường là câu hỏi đòi hỏi suy luận xuyên nhiều phần
    # văn bản (vd. tổng hợp quy định từ nhiều Điều/Chương) mà retrieval theo
    # similarity/từ khoá khó nắm bắt — PageIndex (reasoning-based, vectorless)
    # phù hợp hơn cho những ca này.
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) < threshold "
            f"({score_threshold}). Fallback → PageIndex"
        )
        try:
            return pageindex_search(query, top_k=top_k)
        except Exception as e:
            # PageIndex là API ngoài, có thể lỗi vì lý do nằm ngoài tầm kiểm
            # soát (hết credit, rate limit, mạng...) — "fallback" mà tự crash
            # thì còn tệ hơn không fallback. Degrade về kết quả hybrid (dù
            # dưới ngưỡng) thay vì làm sập toàn bộ retrieval pipeline.
            print(f"  ⚠ PageIndex fallback thất bại ({e}). Dùng kết quả hybrid.")
            return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
