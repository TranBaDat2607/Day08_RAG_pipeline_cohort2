# Image cho backend demo chatbot (FastAPI bọc generate_with_citation — Task 10).
# Đặt ở gốc repo vì Hugging Face Spaces (Docker SDK) mặc định build từ
# Dockerfile tại root.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mã nguồn pipeline (Task 1-10) + dữ liệu chuẩn hoá cần ở runtime:
#   - task6_lexical_search xây BM25 corpus từ data/standardized/*.md
#   - task8_pageindex_vectorless đọc data/standardized/_pageindex_doc_map.json
COPY src/ ./src/
COPY data/standardized/ ./data/standardized/
COPY api/ ./api/

# Hugging Face Spaces (Docker SDK) mặc định expose cổng 7860.
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
