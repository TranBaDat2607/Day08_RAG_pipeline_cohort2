"""
Task 9 — Retrieval Pipeline hoàn chỉnh (group project, backend = FAISS).

Kết hợp semantic search (Task 5, FAISS) + lexical search (Task 6, BM25) + reranking
(Task 7, MMR/RRF) + PageIndex fallback (Task 8) thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search SONG SONG
    2. Merge kết quả bằng RRF (Task 7)
    3. Rerank bằng MMR (Task 7)
    4. Nếu top result score < threshold -> fallback sang PageIndex (Task 8)
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

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold -> fallback PageIndex
DEFAULT_TOP_K = 5
# Task 7 đã chọn MMR làm phương pháp rerank chính (re-order 1 list, tăng diversity,
# cắt trùng lặp - rất hợp văn bản luật/báo nhiều đoạn na ná). RRF được dùng riêng
# ở bước merge dense+sparse bên dưới. Không dùng cross_encoder vì Task 7 không nạp
# reranker model (giữ pipeline nhẹ, nhất quán trên cùng không gian embedding E5).
RERANK_METHOD = "mmr"  # "mmr" | "rrf"


def _safe_search(fn, query: str, top_k: int) -> list[dict]:
    """
    Gọi một retriever (semantic/lexical/pageindex) một cách an toàn.

    Nếu service phụ thuộc chưa sẵn sàng (chưa index, thiếu API key, chưa nạp được
    model...) thì trả [] thay vì raise, để pipeline luôn chạy được và có thể rơi
    xuống nhánh fallback PageIndex.
    """
    try:
        results = fn(query, top_k=top_k)
        return results if isinstance(results, list) else []
    except Exception as e:  # noqa: BLE001 - một retriever lỗi không được làm sập pipeline
        print(f"  [WARN] {getattr(fn, '__name__', 'retriever')} lỗi, bỏ qua: {e}")
        return []


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
          -> Semantic Search -> results_dense
          -> Lexical Search  -> results_sparse
          -> Merge (RRF) -> merged_results
          -> Rerank (MMR) -> reranked_results
          -> If best_score < threshold:
                -> PageIndex Vectorless -> fallback_results

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
    # --- Step 1: Chạy semantic + lexical SONG SONG -------------------------
    # Lấy dư (top_k * 2) ở mỗi nhánh để RRF/MMR có đủ ứng viên để chọn lọc.
    # Mỗi nhánh được bọc _safe_search -> một nhánh hỏng không làm sập cả pipeline.
    fetch_k = max(top_k * 2, top_k)
    with ThreadPoolExecutor(max_workers=2) as ex:
        dense_future = ex.submit(_safe_search, semantic_search, query, fetch_k)
        sparse_future = ex.submit(_safe_search, lexical_search, query, fetch_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # --- Step 2: Merge dense + sparse bằng RRF ------------------------------
    # RRF gộp theo THỨ HẠNG nên không bị lệch do dense/sparse khác thang điểm.
    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    # --- Step 3: Rerank lại danh sách đã merge (MMR) ------------------------
    # MMR re-score theo độ liên quan với query + cắt trùng lặp. Sau bước này
    # 'score' là cosine relevance ([-1, 1]) -> có thể so với score_threshold.
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        # rerank tạo dict mới từ candidate -> 'source' = 'hybrid' được giữ nguyên.
    else:
        final_results = merged[:top_k]

    # --- Step 4: Kiểm tra ngưỡng -> fallback PageIndex ----------------------
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  [WARN] Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback -> PageIndex"
        )
        fallback = _safe_search(pageindex_search, query, top_k)
        if fallback:
            # pageindex_search đã gắn sẵn 'source' = 'pageindex'.
            return fallback[:top_k]
        # PageIndex cũng không có gì -> trả những gì hybrid có (có thể là []).

    # --- Step 5: Trả top_k ------------------------------------------------
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
