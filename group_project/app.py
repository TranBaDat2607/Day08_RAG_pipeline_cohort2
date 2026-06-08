"""
group_project/app.py — FastAPI backend cho RAG Chatbot "LuậtMaTuý AI".

Bọc pipeline của nhóm (Task 9 hybrid retrieval + Task 10 generation có citation)
thành một HTTP endpoint cho giao diện web (group_project/web/) gọi.

Chạy (từ repo root, sau khi đã pip install -r requirements.txt):
    uvicorn group_project.app:app --reload --port 8000

Sau đó mở http://localhost:8000 — backend phục vụ luôn UI tĩnh ở group_project/web/
(cùng origin với API_BASE trong app.js nên không vướng CORS).

Luồng một lượt /chat:
    UI POST { query, session_id?, history? }
      -> (nếu có history/memory) ghép ngữ cảnh hội thoại vào query để hỗ trợ follow-up
      -> generate_with_citation(query)  [Task 10 -> Task 9 hybrid + fallback PageIndex]
      -> trả { answer, sources, retrieval_source }
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core.task10_generation import generate_with_citation

WEB_DIR = Path(__file__).resolve().parent / "web"

# Conversation memory đơn giản, lưu theo session_id ở tầng backend (KHÔNG đổi
# contract Task 9/10). Mỗi entry là list các lượt {"role", "content"}.
SESSIONS: dict[str, list[dict]] = {}
_MAX_TURNS = 6          # số lượt gần nhất giữ lại cho mỗi session
_MAX_HISTORY_CTX = 4    # số lượt đưa vào ngữ cảnh khi rewrite query follow-up


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi động: nạp sẵn FAISS store + embedding model 1 lần (warm-up) để
    request đầu tiên không phải chờ nạp model ~470MB."""
    try:
        from .ingestion.chunk_and_index import get_faiss_store
        from .core.task5_semantic_search import _get_model

        store = get_faiss_store()
        n = store.get("index").ntotal if store.get("index") is not None else 0
        print(f"[startup] FAISS store sẵn sàng: {n} vectors.")
        _get_model()
        print("[startup] Embedding model đã nạp.")
    except Exception as e:  # noqa: BLE001 — warm-up lỗi không nên chặn server khởi động
        print(f"[startup] ⚠ Warm-up lỗi (sẽ nạp lazy ở request đầu): {e}")
    yield


app = FastAPI(title="LuậtMaTuý AI — RAG Backend", lifespan=lifespan)

# Cho phép gọi từ mọi origin (kể cả mở web/index.html bằng file://). Demo nội bộ.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    history: list[dict] | None = None  # [{"role": "user"|"ai", "content": str}, ...]


def _recent_user_turns(session_id: str | None, history: list[dict] | None) -> list[str]:
    """Gộp lịch sử client gửi lên + memory backend, lấy các câu hỏi user gần nhất."""
    turns: list[dict] = []
    if session_id and session_id in SESSIONS:
        turns.extend(SESSIONS[session_id])
    if history:
        turns.extend(history)
    user_msgs = [
        (t.get("content") or "").strip()
        for t in turns
        if (t.get("role") in ("user",)) and (t.get("content") or "").strip()
    ]
    return user_msgs[-_MAX_HISTORY_CTX:]


def _contextualize(query: str, prior_user_msgs: list[str]) -> str:
    """
    Hỗ trợ follow-up: ghép các câu hỏi trước vào query để retrieval hiểu ngữ cảnh
    (vd "thế còn hình phạt thì sao?" cần biết đang nói về tội gì). Giữ nguyên contract
    Task 9/10 — chỉ làm giàu chuỗi query đầu vào.
    """
    if not prior_user_msgs:
        return query
    ctx = "\n".join(f"- {m}" for m in prior_user_msgs)
    return (
        f"Bối cảnh các câu hỏi trước trong hội thoại:\n{ctx}\n\n"
        f"Câu hỏi hiện tại: {query}"
    )


def _shape_sources(sources: list[dict]) -> list[dict]:
    """
    Định dạng sources cho UI: app.js (formatSources) đọc s.source và s.score.
    Lấy 'source' từ metadata để hiển thị tên văn bản nguồn cho người dùng đối chiếu.
    """
    shaped = []
    for s in sources or []:
        meta = s.get("metadata", {}) or {}
        shaped.append({
            "source": meta.get("source") or meta.get("filename") or "Nguồn",
            "type": meta.get("type", "unknown"),
            "score": s.get("score"),
            "content": s.get("content", ""),
        })
    return shaped


@app.get("/health")
def health() -> dict:
    """Readiness check đơn giản."""
    try:
        from .ingestion.chunk_and_index import get_faiss_store

        store = get_faiss_store()
        index = store.get("index")
        n = index.ntotal if index is not None else 0
        return {"status": "ok", "vectors": int(n)}
    except Exception as e:  # noqa: BLE001
        return {"status": "degraded", "error": str(e)}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    """
    Endpoint chat chính. Trả { answer, sources, retrieval_source }.
    """
    query = (req.query or "").strip()
    if not query:
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": [],
            "retrieval_source": "none",
        }

    # Follow-up: ghép ngữ cảnh các câu hỏi trước (client history + memory backend).
    # Bỏ lượt trùng đúng với câu hỏi hiện tại (UI đã thêm nó vào history trước khi gọi).
    prior = [m for m in _recent_user_turns(req.session_id, req.history) if m != query]
    effective_query = _contextualize(query, prior)

    try:
        result = generate_with_citation(effective_query)
    except Exception as e:  # noqa: BLE001 — không để lỗi LLM/retrieval làm sập API
        print(f"[/chat] Lỗi generate: {e}")
        return {
            "answer": "Có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại.",
            "sources": [],
            "retrieval_source": "error",
        }

    answer = result.get("answer", "")

    # Cập nhật memory backend theo session (giữ tối đa _MAX_TURNS lượt gần nhất).
    if req.session_id:
        hist = SESSIONS.setdefault(req.session_id, [])
        hist.append({"role": "user", "content": query})
        hist.append({"role": "ai", "content": answer})
        del hist[:-_MAX_TURNS]

    return {
        "answer": answer,
        "sources": _shape_sources(result.get("sources", [])),
        "retrieval_source": result.get("retrieval_source", "hybrid"),
    }


# Phục vụ UI tĩnh ở group_project/web/ tại "/" — ĐĂNG KÝ SAU các route API ở trên
# để /chat và /health không bị mount "/" che mất.
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("group_project.app:app", host="0.0.0.0", port=8000, reload=True)
