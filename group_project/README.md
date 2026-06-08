# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

### Tổng quan

RAG Chatbot của nhóm **tái sử dụng nguyên vẹn pipeline cá nhân** (Task 1–10 trong
`individual/2A202600778-TranBaDat/src/`) và bọc thêm một lớp giao diện chat. Cùng một bài
toán end-to-end — *thu thập → chuẩn hoá → indexing → hybrid retrieval + vectorless fallback →
generation có citation* — nhưng người dùng tương tác qua trình duyệt thay vì chạy script.

Hệ thống gồm 2 luồng:
- **Offline (ingestion):** chạy 1 lần để dựng kho tri thức (Task 1 → Task 4). Kết quả là một
  vector store **ChromaDB** cục bộ.
- **Online (query):** mỗi lượt hỏi đi qua `web/` → **FastAPI backend** → `retrieve` (Task 9) →
  `generate_with_citation` (Task 10) → trả câu trả lời kèm citation + source về UI.

### Sơ đồ kiến trúc

```
┌──────────────────────────── OFFLINE / INGESTION (chạy 1 lần) ────────────────────────────┐
│                                                                                            │
│  Task 1  thu thập văn bản luật (Playwright @ vbpl.vn)  ─┐                                  │
│  Task 2  crawl báo về nghệ sĩ & ma tuý (Crawl4AI)      ─┤                                  │
│                                                          ▼                                  │
│                                          data/landing/{legal,news}/  (PDF, DOCX, JSON)      │
│                                                          │  Task 3 — MarkItDown             │
│                                                          ▼                                  │
│                                          data/standardized/{legal,news}/*.md                │
│                                                          │  Task 4 — Chunk + Embed + Index  │
│                                                          ▼                                  │
│   RecursiveCharacterTextSplitter (size=1000, overlap=150, tách theo "Chương"/"Điều")        │
│        → embed E5-small (intfloat/multilingual-e5-small, 384d, prefix "passage: ")          │
│        → ChromaDB (persistent, collection "DrugLawDocs", cosine)                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── ONLINE / QUERY (mỗi lượt chat) ──────────────────────────────┐
│                                                                                            │
│   Browser ── group_project/web/ (index.html · app.js · style.css — "LuậtMaTuý AI")         │
│      │  HTTP POST /chat   { "message": "...", "history": [...] }                            │
│      ▼                                                                                      │
│   FastAPI backend  (group_project/app.py — CẦN BUILD)                                       │
│      │                                                                                      │
│      ├─→  retrieve(query)               ── Task 9: src/task9_retrieval_pipeline.py ──┐      │
│      │       ├─→ semantic_search   (Task 5: ChromaDB, E5 "query: ", score=1-distance)│      │
│      │       │            ∥ (song song, ThreadPoolExecutor)                          │      │
│      │       ├─→ lexical_search    (Task 6: BM25Okapi + underthesea, k1=1.5 b=0.75)  │      │
│      │       ├─→ RRF merge         (Task 7: rerank_rrf — gộp dense + sparse)         │      │
│      │       ├─→ MMR rerank        (Task 7: rerank, λ=0.7 — relevance + diversity)   │      │
│      │       └─→ nếu best_score < 0.3 → PageIndex fallback (Task 8, vectorless)      │      │
│      │                                  → source ∈ { "hybrid", "pageindex" }  ───────┘      │
│      │                                                                                      │
│      └─→  generate_with_citation(query)  ── Task 10: src/task10_generation.py ───────┐      │
│              ├─→ reorder_for_llm   (chống "lost in the middle": tốt nhất ở đầu & cuối)│      │
│              ├─→ format_context    (gắn nhãn [Source: ...] cho mỗi chunk)            │      │
│              └─→ OpenAI gpt-4o-mini (temperature=0.3, top_p=0.9, SYSTEM_PROMPT)      │      │
│                                  → { answer, sources, retrieval_source }  ───────────┘      │
│      ▼                                                                                      │
│   JSON response → UI render: câu trả lời + citation [Nguồn, Năm] + danh sách source docs    │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Thành phần & trách nhiệm

| Lớp | Công nghệ | Module / đường dẫn | Vai trò |
|-----|-----------|--------------------|---------|
| Frontend | HTML/CSS/JS thuần | `group_project/web/{index.html,app.js,style.css}` | Giao diện chat "LuậtMaTuý AI"; auth + lịch sử (localStorage); gọi `/chat`, hiển thị answer/citation/source |
| Backend API | FastAPI (Python) | `group_project/app.py` *(cần build)* | Expose `POST /chat`; chuỗi Task 9 → Task 10; giữ **conversation memory** cho follow-up; nạp model 1 lần |
| Retrieval core | — | `src/task9_retrieval_pipeline.py` | `retrieve()` — hybrid (semantic ∥ lexical) + RRF + MMR + fallback ngưỡng `SCORE_THRESHOLD=0.3` |
| Semantic search | sentence-transformers | `src/task5_semantic_search.py` | Dense retrieval trên ChromaDB; E5 prefix `query: ` |
| Lexical search | rank-bm25, underthesea | `src/task6_lexical_search.py` | BM25Okapi; corpus lấy từ chính collection ChromaDB |
| Reranking | tự implement | `src/task7_reranking.py` | `rerank` (MMR) + `rerank_rrf` (gộp nhiều ranked list) |
| Vectorless fallback | PageIndex Cloud | `src/task8_pageindex_vectorless.py` | Reasoning trên cây tài liệu khi hybrid score thấp; `source="pageindex"` |
| Generation | OpenAI `gpt-4o-mini` | `src/task10_generation.py` | `generate_with_citation()` — reorder → format → LLM có citation |
| Vector store | **ChromaDB** (local) | `data/.../chroma_db`, collection `DrugLawDocs` | Lưu embedding 384d, cosine |
| Indexing | langchain-text-splitters, ST | `src/task4_chunking_indexing.py` | Chunk 1000/150, embed E5-small, index |
| Ingestion | Playwright, Crawl4AI, MarkItDown | `src/task1..task3` | Thu thập + chuẩn hoá dữ liệu → markdown |

### Luồng dữ liệu một lượt hỏi

1. **Request:** UI gửi `POST /chat` với `{ message, history }` (history để hỗ trợ follow-up).
2. **Retrieve (Task 9):** backend gọi `retrieve(message, top_k=5)` → chạy semantic + lexical
   song song → RRF merge → MMR rerank. Nếu điểm cao nhất `< 0.3` → fallback PageIndex. Mỗi kết
   quả có `{content, score, metadata, source}` với `source ∈ {hybrid, pageindex}`.
3. **Generate (Task 10):** `generate_with_citation(message)` reorder chunks (chống lost-in-the-
   middle), format context kèm nhãn nguồn, gọi `gpt-4o-mini`. Nếu không có evidence → trả
   *"Tôi không thể xác minh thông tin này từ nguồn hiện có."*
4. **Response:** backend trả `{ answer, sources, retrieval_source }`. UI hiển thị câu trả lời,
   các citation dạng `[Nguồn, Năm]` và danh sách source documents đã dùng.
5. **Follow-up:** backend ghép `history` vào câu hỏi/ngữ cảnh để hỗ trợ hội thoại nhiều lượt
   (conversation memory ở lớp backend, không đổi contract của Task 9/10).

### Tech stack & cấu hình

- **Ngôn ngữ/backend:** Python + FastAPI (gợi ý chạy bằng `uvicorn`).
- **Embedding:** `intfloat/multilingual-e5-small` (384 chiều, bắt buộc prefix `query: `/
  `passage: `, `normalize_embeddings=True`).
- **Vector store:** **ChromaDB** cục bộ (persistent), collection `DrugLawDocs`, không gian cosine.
  > Lưu ý: `requirements.txt` có `weaviate-client` nhưng **vector store đang dùng thực tế là
  > ChromaDB**; Weaviate chỉ là phương án thay thế, không bật.
- **Lexical:** `rank-bm25` (BM25Okapi, k1=1.5, b=0.75) + `underthesea` (tách từ tiếng Việt,
  fallback regex `\w+`).
- **Reranking:** MMR (λ=0.7) cho 1 list; RRF (k=60) để gộp dense + sparse.
- **Vectorless:** PageIndex Cloud (cần `PAGEINDEX_API_KEY`).
- **LLM:** OpenAI `gpt-4o-mini`, `temperature=0.3`, `top_p=0.9` (factual, ít bịa).
- **Secrets (.env):** `OPENAI_API_KEY`, `JINA_API_KEY` (tùy chọn), `PAGEINDEX_API_KEY`,
  `WEAVIATE_URL`/`WEAVIATE_API_KEY` (không dùng nếu chạy ChromaDB).

### Khoảng trống cần hoàn thiện (glue layer)

- [ ] **Dựng index trước:** `python -m src.task4_chunking_indexing` (sau khi có dữ liệu Task 1–3).
- [ ] **Backend `group_project/app.py` (FastAPI):** endpoint `POST /chat` bọc
      `task9.retrieve` + `task10.generate_with_citation`; nạp model/collection 1 lần khi khởi động.
- [ ] **Conversation memory:** lưu lịch sử theo session để hỗ trợ follow-up.
- [ ] **Nối frontend:** sửa `web/app.js` để `fetch('/chat')` thay cho dữ liệu mock localStorage,
      và render `answer` + `sources`.

---

## Phân Công Công Việc

| Thành viên | MSSV       | Nhiệm vụ                                             | Trạng thái |
|----------|------------|------------------------------------------------------|------------|
|Trần Bá Đạt| 2A202600778 | Implement search -> retrival pipeline and merge code | |
|Nguyễn Thành Đạt|2A202600771 | Data collection -> chunking data                     | |
|Nguyễn Thị Bảo Trân|2A202600917| UI/UX + Create a evaluation data set and testing     | |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
