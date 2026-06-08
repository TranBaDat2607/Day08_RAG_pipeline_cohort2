# RAG Evaluation Results

## Framework sử dụng

DeepEval (LLM-judge: gpt-4o-mini)

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (hybrid, no rerank) | Δ (A − B) |
|--------|---|---|---|
| Faithfulness | 0.992 | 0.938 | +0.055 |
| Answer Relevancy | 0.814 | 0.790 | +0.024 |
| Context Recall | 0.812 | 0.812 | +0.000 |
| Context Precision | 0.692 | 0.550 | +0.141 |
| **Average** | **0.828** | **0.773** | **+0.055** |

---

## A/B Comparison Analysis

**Config A (hybrid + rerank):**
> Hybrid retrieval (semantic Weaviate + lexical BM25, gộp bằng RRF), sau đó áp dụng cross-encoder reranking (BGE-reranker-v2-m3) để xếp hạng lại top-k theo mức độ liên quan thực sự với câu hỏi.

**Config B (hybrid, no rerank):**
> Chỉ dùng hybrid retrieval (RRF fusion của semantic + lexical), bỏ qua bước rerank — top-k được lấy trực tiếp theo điểm RRF thô.

**Kết luận:**
> Config A (hybrid + rerank) đạt điểm trung bình cao hơn (0.828 so với 0.773). Điều này phù hợp với kỳ vọng: reranking giúp xếp hạng lại kết quả theo mức độ liên quan ngữ nghĩa thực sự (cross-encoder đọc trực tiếp cặp query-document), khắc phục hạn chế của điểm RRF (chỉ dựa vào thứ hạng thô, chưa 'hiểu' nội dung), từ đó cải thiện chất lượng context đưa vào LLM và độ chính xác của câu trả lời.

---

## Worst Performers (Bottom 3 — theo Config A)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 2... | 1.00 | 0.00 | 0.00 | Retrieval | Context recall thấp — corpus chưa có đủ tài liệu liên quan hoặc retriever bỏ sót evidence quan trọng |
| 2 | Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luậ... | 1.00 | 0.00 | 0.00 | Retrieval | Context recall thấp — corpus chưa có đủ tài liệu liên quan hoặc retriever bỏ sót evidence quan trọng |
| 3 | Trước Nguyễn Công Trí, những nghệ sĩ Việt nào từng bị bắt vì... | 0.88 | 0.60 | 1.00 | Generation | Answer relevance thấp — câu trả lời lạc đề hoặc chưa trả lời thẳng vào câu hỏi |

---

## Recommendations

### Cải tiến 1
**Action:** Bổ sung thêm tài liệu pháp lý/tin tức liên quan vào corpus (đặc biệt các Nghị định/Thông tư hướng dẫn chi tiết) để tăng context recall cho các câu hỏi đòi hỏi tổng hợp từ nhiều nguồn.
**Expected impact:** Tăng Context Recall và Context Precision, giảm tỷ lệ fallback sang PageIndex.

### Cải tiến 2
**Action:** Giữ bước reranking (cross-encoder) làm mặc định trong production — kết quả A/B cho thấy nó cải thiện chất lượng tổng thể so với chỉ dùng RRF thô.
**Expected impact:** Tăng Answer Relevancy và Faithfulness nhờ context đưa vào LLM chính xác hơn.

### Cải tiến 3
**Action:** Với các câu hỏi có Faithfulness thấp, xem xét hạ `temperature` hơn nữa hoặc bổ sung ràng buộc rõ hơn trong system prompt để LLM bám sát evidence, tránh suy diễn ngoài context.
**Expected impact:** Giảm hallucination, tăng độ tin cậy của câu trả lời có trích dẫn.
