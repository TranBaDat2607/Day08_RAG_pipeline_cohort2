"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # cài Playwright browser cho lần đầu
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Danh sách bài báo cần crawl được đọc từ news.json ở thư mục gốc dự án.
NEWS_JSON = Path(__file__).parent.parent / "news.json"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_articles() -> list[dict]:
    """Đọc danh sách bài báo (title, url, source) từ news.json."""
    if not NEWS_JSON.exists():
        raise FileNotFoundError(f"Không tìm thấy {NEWS_JSON}")
    return json.loads(NEWS_JSON.read_text(encoding="utf-8"))


def slugify(text: str, max_len: int = 60) -> str:
    """Chuyển tiêu đề thành slug an toàn để đặt tên file."""
    text = text.lower()
    # bỏ dấu tiếng Việt cơ bản
    text = re.sub(r"[^a-z0-9\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
                  r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:max_len].strip("-") or "article"


async def crawl_article(url: str, title: str = "", source: str = "") -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "source": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        # Lấy markdown (crawl4ai trả về object MarkdownGenerationResult hoặc str)
        markdown = getattr(result, "markdown", "") or ""
        if not isinstance(markdown, str):
            markdown = getattr(markdown, "raw_markdown", "") or str(markdown)

        # Ưu tiên title từ news.json, fallback sang metadata trang
        page_title = title
        if not page_title and getattr(result, "metadata", None):
            page_title = result.metadata.get("title", "Unknown")

        return {
            "url": url,
            "title": page_title or "Unknown",
            "source": source,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": markdown,
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong news.json."""
    setup_directory()
    articles = load_articles()

    for i, item in enumerate(articles, 1):
        url = item["url"]
        title = item.get("title", "")
        source = item.get("source", "")
        print(f"[{i}/{len(articles)}] Crawling: {url}")

        try:
            article = await crawl_article(url, title=title, source=source)
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")
            continue

        if len(article["content_markdown"]) < 200:
            print(f"  ⚠ Nội dung quá ngắn ({len(article['content_markdown'])} ký tự), "
                  "có thể trang chặn crawler.")

        # Đặt tên file theo slug tiêu đề để dễ nhận biết
        slug = slugify(title) if title else f"article-{i:02d}"
        filename = f"{i:02d}-{slug}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {filepath} ({filepath.stat().st_size} bytes)")


if __name__ == "__main__":
    asyncio.run(crawl_all())
