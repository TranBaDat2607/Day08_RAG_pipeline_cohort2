"""
Task 9 - Retrieval Pipeline Hoan Chinh.

Ket hop semantic search + lexical search + reranking + PageIndex fallback
thanh mot pipeline thong nhat.

Logic:
    1. Chay semantic_search + lexical_search song song
    2. Merge ket qua (RRF hoac weighted fusion)
    3. Rerank
    4. Neu top result score < threshold -> fallback sang PageIndex
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

SCORE_THRESHOLD = 0.3   # Neu best score < threshold -> fallback PageIndex
DEFAULT_TOP_K = 5
# Task 7 da chon MMR lam phuong phap rerank chinh (re-order 1 list, tang diversity,
# cat trung lap - rat hop voi van ban luat/bao nhieu doan na na). RRF duoc dung rieng
# o buoc merge dense+sparse ben duoi. Khong dung cross_encoder vi Task 7 khong nap
# reranker model (giu pipeline nhe, nhat quan tren cung khong gian embedding E5).
RERANK_METHOD = "mmr"  # "mmr" | "rrf"


def _safe_search(fn, query: str, top_k: int) -> list[dict]:
    """
    Goi mot retriever (semantic/lexical) mot cach an toan.

    Neu service phu thuoc chua san sang (chua index, thieu Weaviate, thieu API key,
    chua nap duoc model...) thi tra [] thay vi raise, de pipeline luon chay duoc va
    co the roi xuong nhanh fallback PageIndex.
    """
    try:
        results = fn(query, top_k=top_k)
        return results if isinstance(results, list) else []
    except Exception as e:  # noqa: BLE001 - mot retriever loi khong duoc lam sap pipeline
        print(f"  [WARN] {getattr(fn, '__name__', 'retriever')} loi, bo qua: {e}")
        return []


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoan chinh voi fallback logic.

    Pipeline:
        Query
          -> Semantic Search -> results_dense
          -> Lexical Search  -> results_sparse
          -> Merge (RRF) -> merged_results
          -> Rerank -> reranked_results
          -> If best_score < threshold:
                -> PageIndex Vectorless -> fallback_results

    Args:
        query: Cau truy van
        top_k: So luong ket qua cuoi cung
        score_threshold: Nguong diem toi thieu cho hybrid results
        use_reranking: Co ap dung reranking hay khong

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoac 'pageindex'
        }
    """
    # --- Step 1: Chay semantic + lexical SONG SONG -------------------------
    # Lay du (top_k * 2) o moi nhanh de RRF/MMR co du ung vien de chon loc.
    # Moi nhanh duoc boc _safe_search -> mot nhanh hong khong lam sap ca pipeline.
    fetch_k = max(top_k * 2, top_k)
    with ThreadPoolExecutor(max_workers=2) as ex:
        dense_future = ex.submit(_safe_search, semantic_search, query, fetch_k)
        sparse_future = ex.submit(_safe_search, lexical_search, query, fetch_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # --- Step 2: Merge dense + sparse bang RRF ------------------------------
    # RRF gop theo THU HANG nen khong bi lech do dense/sparse khac thang diem.
    merged = rerank_rrf([dense_results, sparse_results], top_k=fetch_k)
    for item in merged:
        item["source"] = "hybrid"

    # --- Step 3: Rerank lai danh sach da merge (MMR) ------------------------
    # MMR re-score theo do lien quan voi query + cat trung lap. Sau buoc nay
    # 'score' la cosine relevance ([-1, 1]) -> co the so voi score_threshold.
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        # rerank tao dict moi tu candidate -> 'source' = 'hybrid' duoc giu nguyen.
    else:
        final_results = merged[:top_k]

    # --- Step 4: Kiem tra nguong -> fallback PageIndex ----------------------
    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  [WARN] Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            f"Fallback -> PageIndex"
        )
        fallback = _safe_search(pageindex_search, query, top_k)
        if fallback:
            # pageindex_search da gan san 'source' = 'pageindex'.
            return fallback[:top_k]
        # PageIndex cung khong co gi -> tra nhung gi hybrid co (co the la []).

    # --- Step 5: Tra top_k ------------------------------------------------
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
