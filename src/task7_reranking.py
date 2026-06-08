"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

=============================================================================
LỰA CHỌN: Cross-encoder reranker — BAAI/bge-reranker-v2-m3
=============================================================================
Lý do:
  1. Chất lượng xếp hạng cao nhất: cross-encoder đưa (query, document) qua
     CÙNG một model để tính điểm liên quan trực tiếp (full cross-attention
     giữa 2 chuỗi), trong khi bi-encoder (dùng ở Task 5) chỉ so cosine giữa
     2 vector embedding tính độc lập — một phép xấp xỉ kém chính xác hơn.
     Với truy vấn pháp luật, độ chính xác top-1/top-3 (đúng Điều luật cần
     tìm) quan trọng hơn tốc độ — cross-encoder đáng giá phần chi phí thêm.
  2. Cùng hệ sinh thái BAAI/BGE với embedding model đã chọn ở Task 4
     (BAAI/bge-m3) — multilingual, được huấn luyện/tối ưu tốt cho tiếng
     Việt và các ngôn ngữ ít tài nguyên, nhất quán trong toàn pipeline.
  3. Chạy local qua `transformers` (AutoModelForSequenceClassification),
     không cần API key (như Jina) — phù hợp dữ liệu pháp luật/tin tức nội
     bộ, tránh chi phí & rủi ro rò rỉ dữ liệu khi gửi qua dịch vụ ngoài.
     (Lưu ý: dùng `transformers` trực tiếp thay vì
     `sentence_transformers.CrossEncoder` để né lỗi import torchcodec/FFmpeg
     đang gặp trong môi trường này — xem Task 4.)

Bên dưới vẫn implement đầy đủ MMR và RRF (được dùng trong các kịch bản khác
của pipeline — vd. RRF để hợp nhất kết quả lexical (Task 6) + semantic
(Task 5) thành hybrid search; MMR để đa dạng hoá kết quả).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Load model + tokenizer một lần và tái sử dụng — tránh tải lại (vài trăm MB)
# cho mỗi lần gọi rerank.
_tokenizer = None
_model = None


def _get_reranker():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        _model.eval()
    return _tokenizer, _model


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    tokenizer, model = _get_reranker()

    # Cross-encoder chấm điểm trực tiếp từng cặp (query, document) — model đọc
    # đồng thời cả 2 chuỗi nên "hiểu" mức độ liên quan sâu hơn nhiều so với so
    # sánh 2 embedding tính riêng lẻ (bi-encoder ở Task 5).
    pairs = [[query, c["content"]] for c in candidates]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = model(**inputs).logits.view(-1).float()
        # bge-reranker xuất ra 1 logit thô (không chặn khoảng) — áp sigmoid để
        # quy về [0,1] như một "relevance probability". Việc này giúp score
        # có cùng thang đo tương đối với cosine similarity ở Task 5, để
        # Task 9 có thể so sánh với 1 ngưỡng (score_threshold) chung khi
        # quyết định fallback sang PageIndex.
        scores = torch.sigmoid(logits).tolist()

    reranked = [
        {**candidate, "score": score}
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked[:top_k]


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
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            # Mức liên quan tới query — khuyến khích chọn doc phù hợp truy vấn
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Mức tương đồng cao nhất với các doc đã chọn — phạt những doc
            # "trùng lặp ý" với kết quả đã có, để tránh top_k toàn các đoạn na
            # ná nhau (vd. nhiều chunk của cùng 1 Điều) và tăng độ đa dạng.
            max_sim_to_selected = max(
                (
                    cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                    for sel_idx in selected
                ),
                default=0.0,
            )

            # lambda_param càng cao -> ưu tiên relevance; càng thấp -> ưu tiên
            # diversity. 0.7 là điểm cân bằng phổ biến: vẫn ưu tiên đúng nội
            # dung liên quan nhưng giảm trùng lặp giữa các kết quả trả về.
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    # Cơ chế: mỗi ranker (vd. semantic search Task 5, lexical search Task 6)
    # cho ra MỘT THỨ HẠNG cho từng document — RRF chỉ dựa vào RANK (vị trí),
    # KHÔNG dùng giá trị score thô. Điều này né được vấn đề các ranker có
    # thang điểm khác nhau, không thể so sánh trực tiếp (cosine similarity
    # của Task 5 nằm trong [0,1] còn BM25 score của Task 6 không có chặn trên
    # cố định). Document được nhiều ranker xếp hạng cao (rank nhỏ) sẽ có
    # tổng 1/(k+rank) lớn -> điểm RRF cao -> ưu tiên lên đầu danh sách hợp
    # nhất. Hằng số k=60 (theo paper Cormack et al. 2009) làm "mượt" chênh
    # lệch giữa các rank thấp (rank 1 và rank 2 không cách biệt quá lớn).
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
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
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
