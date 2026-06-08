"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Ngưỡng ký tự tối thiểu để coi là convert thành công.
# PDF scan (chỉ có ảnh) trả về text rỗng → bỏ qua thay vì ghi file rỗng.
MIN_CONTENT_CHARS = 50


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print(f"  ⚠ Không tìm thấy thư mục: {legal_dir}")
        return

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            # Convert PDF/DOCX sang markdown bằng MarkItDown
            result = md.convert(str(filepath))
            text = result.text_content or ""

            # Một số PDF là bản scan (ảnh) → không trích xuất được text.
            # Bỏ qua, không ghi file rỗng để tránh nhiễu corpus khi index.
            if len(text.strip()) < MIN_CONTENT_CHARS:
                print(f"  ⚠ Bỏ qua (scan/không có text, {len(text.strip())} chars)")
                continue

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print(f"  ⚠ Không tìm thấy thư mục: {news_dir}")
        return

    md = MarkItDown()

    for filepath in sorted(news_dir.iterdir()):
        suffix = filepath.suffix.lower()
        if suffix == ".json":
            print(f"Converting: {filepath.name}")
            # Đọc JSON crawled article, extract content_markdown, lưu thành .md
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            # Thêm metadata header (title, source url, ngày crawl)
            header = f"# {data.get('title', 'Unknown')}\n\n"
            header += f"**Source:** {data.get('url', 'N/A')}\n"
            header += f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"

            # Hỗ trợ nhiều key content khác nhau tuỳ output của crawler
            body = (
                data.get("content_markdown")
                or data.get("markdown")
                or data.get("content")
                or ""
            )
            if len(body.strip()) < MIN_CONTENT_CHARS:
                print(f"  ⚠ Bỏ qua (nội dung rỗng, {len(body.strip())} chars)")
                continue
            output_path.write_text(header + body, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")
        elif suffix in (".html", ".htm"):
            # Nếu crawler lưu HTML thay vì JSON, dùng MarkItDown để convert
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            text = result.text_content or ""
            if len(text.strip()) < MIN_CONTENT_CHARS:
                print(f"  ⚠ Bỏ qua (nội dung rỗng, {len(text.strip())} chars)")
                continue
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
