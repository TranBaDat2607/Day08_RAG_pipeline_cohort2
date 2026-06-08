"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # chunks đã sorted theo score giảm dần (rank 1 = tốt nhất). Tách thành 2
    # nhóm theo vị trí (1-based): lẻ (1, 3, 5, ...) và chẵn (2, 4, 6, ...).
    #   - Nhóm lẻ giữ nguyên thứ tự, đặt ở ĐẦU → rank 1 (tốt nhất) nằm vị trí
    #     đầu tiên, nơi LLM chú ý nhiều nhất.
    #   - Nhóm chẵn đảo ngược thứ tự, đặt ở CUỐI → rank 2 (tốt nhì) rơi xuống
    #     vị trí cuối cùng, nơi LLM chú ý nhiều thứ nhì; các rank kém hơn
    #     (4, 6, ...) bị đẩy vào giữa — đúng với hiệu ứng "lost in the middle"
    #     (LLM nhớ tốt đầu/cuối, dễ bỏ sót phần giữa prompt dài).
    #
    # Ví dụ [1,2,3,4,5]: lẻ=[1,3,5], chẵn=[2,4] → đảo=[4,2]
    #        → kết quả [1,3,5,4,2]  (khớp đúng docstring)
    odd_positions = chunks[0::2]
    even_positions = chunks[1::2]
    even_positions.reverse()

    return odd_positions + even_positions


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        # Khoá "doc_type" (không phải "type") — đúng với schema metadata được
        # tạo ở load_documents()/index_to_vectorstore() (Task 4) và trả về
        # nguyên trạng qua semantic_search/lexical_search/pageindex_search.
        doc_type = metadata.get("doc_type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve — hybrid (semantic + lexical + rerank), tự fallback
    # PageIndex nếu điểm thấp (logic đầy đủ ở Task 9).
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có",
            "sources": [],
            "retrieval_source": "none",
        }

    # Step 2: Reorder để chunk quan trọng nhất nằm ở đầu/cuối context — giảm
    # rủi ro LLM "bỏ sót" chunk quan trọng nằm giữa prompt dài.
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context kèm label nguồn để LLM trích dẫn đúng theo
    # [Document N | Source ...] — khớp với yêu cầu citation trong SYSTEM_PROMPT.
    context = format_context(reordered)

    # Step 4: Build prompt — tách rõ phần Context và Question để LLM phân biệt
    # "evidence được cung cấp" với "câu hỏi cần trả lời", tránh nhầm lẫn khi
    # trích dẫn.
    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    # Step 5: Gọi LLM — gpt-4o-mini: đủ mạnh để tổng hợp + trích dẫn chính xác
    # với chi phí/độ trễ thấp, phù hợp tác vụ RAG factual (không cần model lớn
    # nhất). temperature=0.3 & top_p=0.9 (xem giải thích ở phần CONFIGURATION)
    # ưu tiên output bám sát evidence, hạn chế "sáng tạo"/hallucination.
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content

    # Step 6: Trả về answer kèm sources gốc (thứ tự theo score, không phải
    # thứ tự đã reorder) để người dùng/kiểm thử dễ đối chiếu độ liên quan thật.
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
