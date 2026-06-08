"""
Task 8 — PageIndex Vectorless RAG (group project).

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà KHÔNG cần vector store / embedding: tài liệu được
PageIndex phân tích thành một "cây" cấu trúc (mục lục → điều/khoản), và việc
truy hồi được thực hiện bằng cách cho LLM "duyệt cây" để chọn ra các node liên
quan (reasoning-based retrieval) thay vì so khớp vector. Rất hợp với văn bản
pháp luật vốn có cấu trúc Chương/Điều/Khoản rõ ràng.

Cài đặt:
    pip install pageindex

Dùng làm FALLBACK trong Task 9 khi hybrid search (dense+sparse) cho điểm thấp.
Nếu thiếu PAGEINDEX_API_KEY hoặc chưa upload tài liệu, pageindex_search() trả []
một cách an toàn (Task 9 bọc _safe_search) -> pipeline vẫn chạy offline được.

Luồng hoạt động (PageIndex Cloud API):
    1. submit_document(pdf)  -> trả về doc_id ngay, xử lý (OCR + dựng cây) chạy nền
    2. is_retrieval_ready(doc_id) -> poll tới khi tài liệu sẵn sàng truy hồi
    3. submit_query(doc_id, query) -> retrieval_id
    4. get_retrieval(retrieval_id) -> các node liên quan (retrieved_nodes)
"""

import os
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

# core/ -> group_project/ -> repo root. Dữ liệu nằm ở group_project/data/ (FAISS,
# landing, standardized của nhóm); .env đặt ở repo root (theo CLAUDE.md).
GROUP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = GROUP_DIR.parent
load_dotenv()                                    # tìm .env từ cwd trở lên
load_dotenv(REPO_ROOT / ".env", override=True)   # .env ở repo root
load_dotenv(GROUP_DIR / ".env", override=True)   # ưu tiên group_project/.env (nơi nhóm để key)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
DATA_DIR = GROUP_DIR / "data"
STANDARDIZED_DIR = DATA_DIR / "standardized"
LANDING_DIR = DATA_DIR / "landing"

# Registry để không phải upload lại mỗi lần chạy: {filename: {"doc_id", "type"}}
REGISTRY_PATH = DATA_DIR / "pageindex_registry.json"

# PageIndex tài liệu pháp luật là PDF có cấu trúc -> nguồn lý tưởng cho PageIndex.
LEGAL_PDF_DIR = LANDING_DIR / "legal"

# Giới hạn thời gian chờ (giây)
_READY_TIMEOUT = 240      # chờ tài liệu xử lý xong
_RETRIEVAL_TIMEOUT = 60   # chờ một lượt truy hồi


def _get_client():
    """Khởi tạo PageIndexClient từ API key trong .env."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "Thiếu PAGEINDEX_API_KEY trong .env. Đăng ký tại https://pageindex.ai/"
        )
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        try:
            return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_documents(wait_ready: bool = True) -> dict:
    """
    Upload toàn bộ văn bản pháp luật (PDF) lên PageIndex.

    - Tự bỏ qua các file đã upload trước đó (đối chiếu theo tên với
      list_documents trên server + registry cục bộ) để tránh upload trùng.
    - Lưu mapping {filename: {"doc_id", "type"}} vào data/pageindex_registry.json
      để pageindex_search() dùng lại mà không phải upload lại.

    Args:
        wait_ready: nếu True, chờ tới khi các tài liệu sẵn sàng truy hồi.

    Returns:
        registry dict đã cập nhật.
    """
    client = _get_client()
    registry = _load_registry()

    # Lập map tên-file -> doc_id từ những gì đã có trên server (tránh upload lại).
    server_by_name = {}
    try:
        listed = client.list_documents(limit=100)
        for d in listed.get("documents", []):
            name = d.get("name")
            if name:
                server_by_name[name] = d.get("id")
    except Exception as e:  # noqa: BLE001 — list lỗi không nên chặn upload
        print(f"  ⚠ Không list được documents: {e}")

    pdfs = sorted(p for p in LEGAL_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"⚠ Không tìm thấy PDF nào trong {LEGAL_PDF_DIR}")
        return registry

    for pdf in pdfs:
        name = pdf.name
        if name in registry and registry[name].get("doc_id"):
            print(f"  ↩ Đã có trong registry, bỏ qua: {name}")
            continue
        if name in server_by_name:
            registry[name] = {"doc_id": server_by_name[name], "type": "legal"}
            print(f"  ↩ Đã có trên PageIndex, dùng lại: {name}")
            continue
        try:
            resp = client.submit_document(str(pdf))
            doc_id = resp["doc_id"]
            registry[name] = {"doc_id": doc_id, "type": "legal"}
            print(f"  ✓ Uploaded: {name} -> {doc_id}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ Lỗi upload {name}: {e}")

        _save_registry(registry)  # lưu dần để không mất tiến độ nếu gián đoạn

    if wait_ready:
        print("Chờ tài liệu xử lý xong (OCR + dựng cây)...")
        deadline = time.time() + _READY_TIMEOUT
        pending = {info["doc_id"] for info in registry.values() if info.get("doc_id")}
        while pending and time.time() < deadline:
            for doc_id in list(pending):
                try:
                    if client.is_retrieval_ready(doc_id):
                        pending.discard(doc_id)
                except Exception:  # noqa: BLE001
                    pending.discard(doc_id)
            if pending:
                time.sleep(5)
        if pending:
            print(f"  ⚠ {len(pending)} tài liệu chưa sẵn sàng (sẽ sẵn sàng sau).")
        else:
            print("  ✓ Tất cả tài liệu đã sẵn sàng.")

    _save_registry(registry)
    return registry


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def _extract_node_text(node: dict) -> str:
    """Gộp các đoạn 'relevant_content' của một node thành một chuỗi."""
    parts = []
    contents = node.get("relevant_contents") or []
    # relevant_contents có thể là list lồng list -> duyệt phẳng.
    stack = list(contents)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack[:0] = item  # mở rộng list lồng nhau
        elif isinstance(item, dict):
            txt = item.get("relevant_content") or item.get("content") or ""
            if txt:
                parts.append(txt.strip())
        elif isinstance(item, str):
            parts.append(item.strip())
    if not parts:
        # một số node chỉ có summary/title
        for k in ("text", "content", "summary"):
            if node.get(k):
                parts.append(str(node[k]))
                break
    return "\n\n".join(p for p in parts if p)


def _node_filename(node: dict, fallback: str) -> str:
    """Lấy tên file nguồn từ metadata của node (metadata[1]) nếu có."""
    meta = node.get("metadata")
    if isinstance(meta, list) and len(meta) > 1 and meta[1]:
        return meta[1]
    if isinstance(meta, dict):
        return meta.get("filename") or meta.get("source") or fallback
    return fallback


def _query_single_doc(client, doc_id: str, filename: str, query: str) -> list[dict]:
    """Truy hồi trên 1 tài liệu, trả về list node đã chuẩn hoá (chưa cắt top_k)."""
    out: list[dict] = []
    try:
        sub = client.submit_query(doc_id, query)
        retrieval_id = sub["retrieval_id"]
    except Exception:  # noqa: BLE001
        return out

    deadline = time.time() + _RETRIEVAL_TIMEOUT
    res = {}
    while time.time() < deadline:
        try:
            res = client.get_retrieval(retrieval_id)
        except Exception:  # noqa: BLE001
            break
        if res.get("status") == "completed":
            break
        if res.get("status") == "failed":
            return out
        time.sleep(2)

    nodes = res.get("retrieved_nodes") or []
    n = len(nodes)
    for rank, node in enumerate(nodes):
        content = _extract_node_text(node)
        if not content:
            continue
        fname = _node_filename(node, filename)
        out.append(
            {
                "content": content,
                # PageIndex không trả score; gán theo thứ hạng (node đầu = liên quan nhất).
                "score": round(1.0 / (rank + 1), 6),
                "metadata": {
                    "source": fname,
                    "filename": fname,
                    "title": node.get("title", ""),
                    "node_id": node.get("id") or node.get("node_id", ""),
                    "doc_id": doc_id,
                    "type": "legal",
                    "_rank": rank,
                    "_n": n,
                },
                "source": "pageindex",
            }
        )
    return out


def _chat_fallback(client, query: str, doc_ids: list[str], top_k: int) -> list[dict]:
    """
    Fallback khi retrieval (deprecated) trả rỗng: dùng chat-completions có
    citation, rồi rút trích các đoạn được trích dẫn làm 'chunks'.
    """
    if not doc_ids:
        return []
    try:
        resp = client.chat_completions(
            messages=[{"role": "user", "content": query}],
            doc_id=doc_ids[:10],  # API nhận tối đa một danh sách doc_id
            enable_citations=True,
        )
    except Exception:  # noqa: BLE001
        return []

    results: list[dict] = []
    try:
        choice = resp["choices"][0]
        msg = choice.get("message", {})
        # Một số phiên bản trả 'citations' kèm đoạn nguồn.
        citations = msg.get("citations") or choice.get("citations") or []
        for rank, c in enumerate(citations[:top_k]):
            text = (
                c.get("content")
                or c.get("text")
                or c.get("relevant_content")
                or ""
            )
            if not text:
                continue
            results.append(
                {
                    "content": text.strip(),
                    "score": round(1.0 / (rank + 1), 6),
                    "metadata": {
                        "source": c.get("filename") or c.get("doc_id") or "pageindex",
                        "title": c.get("title", ""),
                        "type": "legal",
                    },
                    "source": "pageindex",
                }
            )
        # Không có citation có cấu trúc -> dùng nguyên câu trả lời làm 1 chunk.
        if not results and msg.get("content"):
            results.append(
                {
                    "content": msg["content"].strip(),
                    "score": 1.0,
                    "metadata": {"source": "pageindex-chat", "type": "legal"},
                    "source": "pageindex",
                }
            )
    except (KeyError, IndexError, TypeError):
        return []
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt (xem Task 9).

    Truy hồi song song trên tất cả tài liệu trong registry, gộp các node liên
    quan, sắp xếp theo score giảm dần và trả về top_k.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,   # có 'source' = tên file để Task 10 trích dẫn
            'source': 'pageindex'
        }
    """
    registry = _load_registry()
    if not registry:
        # Chưa upload tài liệu nào -> không có gì để truy hồi.
        return []

    client = _get_client()
    docs = [
        (info["doc_id"], fname)
        for fname, info in registry.items()
        if info.get("doc_id")
    ]
    if not docs:
        return []

    collected: list[dict] = []
    # Mỗi lượt truy hồi là async + poll -> chạy song song cho nhanh.
    max_workers = min(8, len(docs))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_query_single_doc, client, doc_id, fname, query): fname
            for doc_id, fname in docs
        }
        for fut in as_completed(futures):
            try:
                collected.extend(fut.result())
            except Exception:  # noqa: BLE001
                continue

    # Fallback sang chat-completions nếu retrieval không ra gì.
    if not collected:
        collected = _chat_fallback(client, query, [d for d, _ in docs], top_k)

    collected.sort(key=lambda r: r["score"], reverse=True)
    return collected[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query: 'Các hành vi bị nghiêm cấm về ma túy'")
        results = pageindex_search("Các hành vi bị nghiêm cấm về ma túy", top_k=3)
        if not results:
            print("  (không có kết quả — kiểm tra registry/upload)")
        for r in results:
            src = r["metadata"].get("source", "?")
            print(f"[{r['score']:.3f}] ({src}) {r['content'][:120]}...")
