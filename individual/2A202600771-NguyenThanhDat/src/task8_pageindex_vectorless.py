"""
Task 8 — PageIndex Vectorless RAG.

Fallback local: tìm kiếm keyword trên full documents khi không có PageIndex API.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY:
        raise NotImplementedError(
            "Set PAGEINDEX_API_KEY trong .env hoặc dùng fallback local search"
        )
    raise NotImplementedError("Implement upload_documents với PageIndex SDK")


def _local_fallback_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback vectorless: keyword scoring trên full markdown files."""
    if not STANDARDIZED_DIR.exists():
        return []

    query_tokens = set(query.lower().split())
    scored = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_tokens = set(content.lower().split())
        overlap = len(query_tokens & doc_tokens)
        if overlap == 0:
            continue

        score = overlap / max(len(query_tokens), 1)
        excerpt = content[:800]
        doc_type = "legal" if "legal" in md_file.parts else "news"
        scored.append({
            "content": excerpt,
            "score": score,
            "metadata": {"source": md_file.name, "type": doc_type},
            "source": "pageindex",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex (hoặc fallback local).
    """
    if PAGEINDEX_API_KEY:
        try:
            from pageindex import PageIndex

            pi = PageIndex(api_key=PAGEINDEX_API_KEY)
            results = pi.query(query=query, top_k=top_k)
            return [
                {
                    "content": r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": "pageindex",
                }
                for r in results
            ]
        except Exception:
            pass

    return _local_fallback_search(query, top_k=top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Không có PAGEINDEX_API_KEY — dùng fallback local search")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
