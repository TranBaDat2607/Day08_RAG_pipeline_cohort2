"""
Task 7 — Reranking Module.

PHƯƠNG PHÁP ĐÃ CHỌN: **MMR (Maximal Marginal Relevance)** — tự implement, KHÔNG
dùng cross-encoder model.

Vì sao MMR (so với RRF) cho bài này:
    - Contract của Task 7 là rerank(query, candidates, top_k): re-order MỘT danh
      sách ứng viên duy nhất. MMR đúng bản chất việc này — nó vừa giữ độ liên quan
      với query, vừa loại bớt các chunk gần-trùng-lặp để tăng diversity.
    - RRF (Reciprocal Rank Fusion) là để GỘP NHIỀU ranked list (dense + sparse) →
      đó là việc của Task 9, không phải rerank một list. Vì vậy RRF được giữ ở
      hàm rerank_rrf() riêng cho Task 9 gọi, không phải entry point chính.
    - Dữ liệu tiếng Việt ở đây (văn bản luật + báo) có rất nhiều đoạn lặp/boilerplate
      (điều khoản, mở bài báo na ná nhau). MMR cắt trùng lặp → context đưa vào LLM
      đa dạng và bao phủ tốt hơn, đặc biệt hữu ích cho generation ở Task 10.

MMR không cần "reranker model" — nó tái sử dụng CHÍNH embedding model E5 đã chọn ở
Task 4 (đồng bộ không gian vector, đồng bộ prefix query/passage), nên nhất quán với
toàn pipeline và không thêm dependency mới.

Công thức:
    MMR = λ * sim(query, doc) - (1-λ) * max_{d∈selected} sim(doc, d)
    λ=1.0 → thuần relevance; λ=0.0 → thuần diversity. Mặc định 0.7 (nghiêng relevance).
"""

import math
from functools import lru_cache

from src.task4_chunking_indexing import (
    EMBED_PASSAGE_PREFIX,
    EMBED_QUERY_PREFIX,
    normalize_text,
)


# =============================================================================
# Helpers: embedding (tái dùng model E5 đã cache ở Task 5) + cosine similarity
# =============================================================================

@lru_cache(maxsize=1)
def _get_model():
    """Dùng lại đúng instance model đã cache ở Task 5 (tránh nạp lại ~470MB)."""
    from src.task5_semantic_search import _get_model as _task5_model

    return _task5_model()


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector. An toàn với vector 0 (trả 0.0)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_query(query: str) -> list[float]:
    """Embed query đúng quy ước E5: prefix 'query: ' + normalize_embeddings."""
    model = _get_model()
    return model.encode(
        EMBED_QUERY_PREFIX + normalize_text(query),
        normalize_embeddings=True,
    ).tolist()


def _ensure_candidate_embeddings(candidates: list[dict]) -> list[list[float]]:
    """
    Lấy embedding cho từng candidate.

    - Nếu candidate đã có sẵn key 'embedding' (vd Task 9 truyền xuống) → dùng lại.
    - Nếu chưa có → embed từ 'content' bằng prefix 'passage: ' (đồng bộ Task 4).
    Embed theo batch cho phần thiếu để nhanh.
    """
    embeddings: list[list[float]] = [None] * len(candidates)
    to_embed_idx, to_embed_text = [], []

    for i, c in enumerate(candidates):
        emb = c.get("embedding")
        if emb is not None:
            embeddings[i] = list(emb)
        else:
            to_embed_idx.append(i)
            to_embed_text.append(EMBED_PASSAGE_PREFIX + normalize_text(c.get("content", "")))

    if to_embed_text:
        model = _get_model()
        vecs = model.encode(to_embed_text, normalize_embeddings=True)
        for i, v in zip(to_embed_idx, vecs):
            embeddings[i] = v.tolist()

    return embeddings


# =============================================================================
# MMR — phương pháp reranking chính của Task 7
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query (cùng không gian E5).
        candidates: List of {'content': str, 'score': float, 'metadata': dict,
                    'embedding'?: list}. Thiếu 'embedding' sẽ tự embed từ 'content'.
        top_k: Số lượng kết quả.
        lambda_param: Trade-off relevance (1.0) ↔ diversity (0.0).

    Returns:
        List of ≤ top_k candidates theo đúng thứ tự MMR đã chọn (thứ tự CHÍNH là
        kết quả của rerank). Mỗi item gắn 'score' (= relevance với query) và
        'mmr_rank' (thứ hạng do MMR quyết định).

    Có thể gọi trực tiếp với query_embedding sẵn có (vd từ Task 9), hoặc để trống
    'embedding' trong candidates và hàm sẽ tự embed từ 'content'.
    """
    if not candidates:
        return []
    embeddings = _ensure_candidate_embeddings(candidates)
    return _mmr_select(query_embedding, candidates, embeddings, top_k, lambda_param)


def _mmr_select(
    query_embedding: list[float],
    candidates: list[dict],
    embeddings: list[list[float]],
    top_k: int,
    lambda_param: float,
) -> list[dict]:
    relevance = [cosine_sim(query_embedding, emb) for emb in embeddings]
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_mmr = None, float("-inf")
        for idx in remaining:
            max_sim = 0.0
            for sel_idx in selected:
                sim = cosine_sim(embeddings[idx], embeddings[sel_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_param * relevance[idx] - (1.0 - lambda_param) * max_sim
            if mmr > best_mmr:
                best_mmr, best_idx = mmr, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    out = []
    for rank, idx in enumerate(selected):
        item = dict(candidates[idx])
        item["score"] = relevance[idx]      # điểm relevance gốc (dễ diễn giải, [-1,1])
        item["mmr_rank"] = rank             # thứ hạng do MMR quyết định
        out.append(item)
    return out


# =============================================================================
# RRF — GIỮ LẠI cho Task 9 (gộp NHIỀU ranked list dense+sparse), không phải
# entry point của Task 7. Ở đây để pipeline Task 9 import dùng trực tiếp.
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

        RRF(d) = Σ_r 1 / (k + rank_r(d))

    Args:
        ranked_lists: Danh sách các ranked list (mỗi list từ 1 ranker).
        top_k: Số kết quả cuối.
        k: Hằng số làm mượt (mặc định 60, theo Cormack et al. 2009).

    Returns:
        List ≤ top_k, sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = dict(content_map[content])
        item["score"] = score
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "mmr" (mặc định) | "rrf"
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface. Mặc định dùng MMR (lựa chọn của Task 7).

    Args:
        query: Câu truy vấn.
        candidates: Danh sách candidates từ retrieval
                    ({'content','score','metadata', 'embedding'?}).
        top_k: Số kết quả sau rerank.
        method: "mmr" (re-order 1 list) hoặc "rrf" (gộp nhiều list — dành Task 9).
        lambda_param: Trade-off relevance↔diversity cho MMR.

    Returns:
        List ≤ top_k candidates đã rerank.
    """
    if method == "mmr":
        if not candidates:
            return []
        query_embedding = _embed_query(query)
        embeddings = _ensure_candidate_embeddings(candidates)
        return _mmr_select(query_embedding, candidates, embeddings, top_k, lambda_param)
    elif method == "rrf":
        # RRF cần nhiều ranked list — gọi rerank_rrf(ranked_lists) trực tiếp (Task 9).
        raise ValueError(
            "RRF gộp nhiều ranked list: gọi rerank_rrf(ranked_lists, top_k) trực tiếp."
        )
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Demo: 2 chunk đầu gần-trùng-lặp về 'tàng trữ' — MMR nên đẩy 1 cái xuống để
    # nhường chỗ cho chunk khác chủ đề, tăng diversity.
    dummy_candidates = [
        {"content": "Điều 249: Tội tàng trữ trái phép chất ma tuý bị phạt tù.", "score": 0.81, "metadata": {"source": "blhs"}},
        {"content": "Tội tàng trữ trái phép chất ma tuý có thể bị phạt tù theo điều 249.", "score": 0.80, "metadata": {"source": "blhs"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý tại quán bar.", "score": 0.62, "metadata": {"source": "news"}},
        {"content": "Luật Phòng, chống ma tuý 2021 quy định về cai nghiện tự nguyện.", "score": 0.55, "metadata": {"source": "luat-2021"}},
    ]
    print("== MMR rerank (λ=0.7) ==")
    for r in rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=3, method="mmr"):
        print(f"[score={r['score']:.3f} | mmr_rank={r['mmr_rank']}] {r['content']}")
