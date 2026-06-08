"""Task 3 — Convert landing files sang Markdown (group project)."""

import json
import re
import unicodedata
from pathlib import Path

from markitdown import MarkItDown

from .config import LANDING_DIR, STANDARDIZED_DIR


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def convert_legal_docs() -> int:
    legal_dir = LANDING_DIR / "legal"
    output_dir = STANDARDIZED_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Không có thư mục landing/legal/")
        return 0

    md = MarkItDown()
    count = 0
    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue
        print(f"Converting: {filepath.name}")
        result = md.convert(str(filepath))
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(normalize_text(result.text_content), encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name}")
        count += 1
    return count


def convert_news_articles() -> int:
    news_dir = LANDING_DIR / "news"
    output_dir = STANDARDIZED_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Không có thư mục landing/news/")
        return 0

    count = 0
    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() != ".json":
            continue
        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        header = f"# {data.get('title', 'Unknown')}\n\n"
        header += f"**Source:** {data.get('url', 'N/A')}\n"
        header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        body = data.get("content_markdown", data.get("content", ""))
        output_path.write_text(normalize_text(header + body), encoding="utf-8")
        print(f"  ✓ Saved: {output_path.name}")
        count += 1
    return count


def convert_all() -> tuple[int, int]:
    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()
    print("\n--- News Articles ---")
    news_count = convert_news_articles()
    print(f"\n✓ Converted {legal_count} legal + {news_count} news → {STANDARDIZED_DIR}")
    return legal_count, news_count
