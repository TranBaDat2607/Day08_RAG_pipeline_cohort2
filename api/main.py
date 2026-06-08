"""
FastAPI backend cho demo chatbot — bọc generate_with_citation() (Task 10)
thành một REST endpoint để web UI tĩnh (GitHub Pages) gọi tới qua HTTP.
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.task10_generation import generate_with_citation

app = FastAPI(title="LuậtMaTuý AI — Chat API")

# CORS: web UI host trên domain khác (GitHub Pages) cần được phép gọi API này
# từ trình duyệt. Đọc danh sách origin từ env để đổi domain khi deploy mà
# không cần sửa code; mặc định "*" để tiện chạy/test cục bộ.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    result = generate_with_citation(req.query)
    return {
        "answer": result["answer"],
        # Chỉ trả về phần thiết yếu cho UI hiển thị citation (source/doc_type/
        # score) — tránh gửi nguyên văn nội dung chunk (có thể dài) qua API.
        "sources": [
            {
                "source": chunk.get("metadata", {}).get("source", ""),
                "doc_type": chunk.get("metadata", {}).get("doc_type", ""),
                "score": chunk.get("score"),
            }
            for chunk in result["sources"]
        ],
        "retrieval_source": result["retrieval_source"],
    }
