# Hướng Dẫn Chạy — RAG Chatbot "LuậtMaTuý AI"

Hướng dẫn chạy chatbot RAG của nhóm: backend FastAPI (bọc pipeline Task 9 retrieval +
Task 10 generation) phục vụ luôn giao diện web. **Tất cả lệnh chạy từ thư mục gốc repo**
(`Day08_RAG_pipeline_cohort2/`), không phải từ trong `group_project/`.

---

## 1. Yêu cầu

- Python 3.10+ (đang dùng conda env `vinai`, Python 3.13).
- Đã có sẵn FAISS index ở `group_project/data/faiss/` (Nguyễn Thành Đạt đã dựng).
  Kiểm tra nhanh: thư mục `group_project/data/faiss/` có `index.faiss` và `metadata.pkl`.
- Một **OpenAI API key hợp lệ** (cho Task 10 — sinh câu trả lời).

---

## 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> Pipeline cần: `faiss-cpu`, `sentence-transformers`, `rank-bm25`, `langchain-text-splitters`,
> `openai`, `python-dotenv`, `fastapi`, `uvicorn`. (`underthesea` là tuỳ chọn — không có thì
> Task 6 tự fallback sang tokenizer regex.)

---

## 3. Cấu hình API key

Mở file **`group_project/.env`** và điền key của bạn:

```env
# BẮT BUỘC — để chatbot sinh câu trả lời
OPENAI_API_KEY=sk-...

# Tuỳ chọn — fallback vectorless (KHÔNG cần khi dùng data cục bộ)
PAGEINDEX_API_KEY=

# Không dùng (Task 7 dùng MMR, không dùng Jina)
JINA_API_KEY=
```

Lưu ý:
- File `.env` đã được **git-ignore**, key của bạn sẽ không bị commit.
- Chỉ **`OPENAI_API_KEY` là bắt buộc**. `PAGEINDEX_API_KEY` chỉ dùng cho nhánh fallback
  (khi điểm hybrid < 0.3); với dữ liệu cục bộ trong `data/` thì không cần.
- Giá trị trong `.env` sẽ **ghi đè** biến môi trường cũ trong shell (`override=True`), nên
  nếu trước đó bạn lỡ set một key hết hạn ở môi trường, key trong `.env` vẫn được ưu tiên.

---

## 4. Chạy chatbot (backend + UI)

```bash
uvicorn group_project.app:app --reload --port 8000
```

Đợi log hiện `[startup] FAISS store sẵn sàng: ... vectors.` và `[startup] Embedding model đã nạp.`
(lần đầu mất vài giây để nạp model embedding ~470MB), rồi mở trình duyệt:

> ### 👉 http://localhost:8000

**KHÔNG** mở trực tiếp `group_project/web/index.html` bằng cách nhấp đúp file — giao diện gọi
API ở `http://localhost:8000/chat`, nên backend phải đang chạy. Cứ mở URL trên là backend
phục vụ luôn giao diện (cùng origin, không vướng CORS).

### Đăng nhập (tài khoản demo)
| Email | Mật khẩu | Vai trò |
|-------|----------|---------|
| `admin@luatmatuy.vn` | `admin123` | Quản trị viên |
| `nguyen@phapluat.vn` | `demo123` | Người dùng |

Sau khi đăng nhập: bấm một "Chủ đề nhanh" hoặc gõ câu hỏi, ví dụ
*"Mức phạt tù cho tội tàng trữ ma tuý theo BLHS 2015?"*. Câu trả lời sẽ kèm **citation**
`[Nguồn]` và danh sách **source documents**. Hỏi tiếp (follow-up) sẽ nhớ ngữ cảnh trước đó.

> 💡 Khi demo, có thể bỏ `--reload` để server không tự khởi động lại (mỗi lần restart phải
> nạp lại model). Dùng `--reload` khi đang chỉnh code.

---

## 5. Kiểm tra nhanh bằng API (không cần UI)

```bash
# Health check
curl http://localhost:8000/health
# -> {"status":"ok","vectors":360}

# Hỏi thử
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hình phạt tội tàng trữ trái phép chất ma tuý?"}'
# -> { "answer": "...", "sources": [{"source","score",...}], "retrieval_source": "hybrid" }
```

Hợp đồng (contract) của `POST /chat`:
- **Request:** `{ "query": str, "session_id"?: str, "history"?: [{"role","content"}] }`
- **Response:** `{ "answer": str, "sources": [{source, type, score, content}], "retrieval_source": "hybrid"|"pageindex"|"none" }`

---

## 6. Chạy thử từng module pipeline (gỡ lỗi)

```bash
python -m group_project.core.task5_semantic_search     # dense retrieval (FAISS)
python -m group_project.core.task6_lexical_search      # BM25 (+ TF-IDF bonus)
python -m group_project.core.task9_retrieval_pipeline  # hybrid + fallback (KHÔNG cần API key)
python -m group_project.core.task10_generation         # full RAG generation (CẦN OPENAI_API_KEY)
```

`task5/6/9` chạy hoàn toàn cục bộ trên `data/faiss/`, không cần API key — tiện để xác minh
retrieval trước khi cắm key OpenAI.

---

## 7. Dựng lại index (nếu cần)

FAISS index đã có sẵn nên **không cần làm bước này**. Chỉ chạy lại khi index trống/đổi dữ liệu:

```bash
# Copy data từ bài cá nhân rồi build lại FAISS
python -m group_project.ingestion.run_pipeline --copy-from-individual

# Chỉ rebuild FAISS (đã có landing + standardized)
python -m group_project.ingestion.run_pipeline --skip-collect --skip-crawl --skip-convert
```

---

## 8. Lỗi thường gặp

| Triệu chứng | Nguyên nhân & cách xử lý |
|-------------|--------------------------|
| UI báo *"Không thể kết nối đến hệ thống AI"* | Backend chưa chạy, hoặc đang mở `index.html` bằng `file://`. Chạy `uvicorn ...` rồi mở `http://localhost:8000`. |
| Trả về *"Có lỗi xảy ra khi xử lý câu hỏi"* | Thường do `OPENAI_API_KEY` sai/hết hạn (lỗi 401). Kiểm tra lại key trong `group_project/.env`. |
| Trả về *"Tôi không thể xác minh thông tin này..."* | Retrieval không tìm được evidence (FAISS rỗng / câu hỏi ngoài phạm vi dữ liệu). Kiểm tra `data/faiss/` và thử câu hỏi về luật/ma tuý. |
| `ModuleNotFoundError: faiss` (hoặc `fastapi`) | Chưa cài dependency: `pip install -r requirements.txt`. |
| Cổng 8000 đang bận | Đổi cổng: `uvicorn group_project.app:app --port 8010` (nhớ sửa `API_BASE` trong `web/app.js` nếu khác 8000). |
