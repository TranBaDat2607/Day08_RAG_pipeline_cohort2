"""Task 2 — Crawl bài báo về nghệ sĩ liên quan ma tuý (group project)."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import LANDING_DIR

DATA_DIR = LANDING_DIR / "news"
MIN_CONTENT_SIZE = 500

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

ARTICLE_URLS = [
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://ngoisao.vn/theo-dong-su-kien/kien-thuc/be-boi-loat-sao-viet-dinh-chat-cam-ma-tuy-da-ton-tai-bao-lau-trong-co-the-va-cach-phat-hien-chuan-xac-nhat-d485348.html",
    "https://baoquangninh.vn/hau-qua-nghiem-trong-khi-nghe-si-viet-lien-tuc-vuong-on-ao-ma-tuy-3407229.html",
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo",
    "https://vietnamnet.vn/loat-ca-si-dinh-chat-cam-ma-tuy-pha-huy-nao-bo-nguoi-tre-ra-sao-2518285.html",
]


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    selectors = [
        "article",
        ".article-content",
        ".detail-content",
        ".content-detail",
        ".fck_detail",
        ".main-content",
        "#maincontent",
        ".cms-body",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            text = node.get_text("\n", strip=True)
            if len(text) > 200:
                return title, text

    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    return title, "\n\n".join(paragraphs)


def _crawl_with_requests(url: str) -> dict:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"

    title, content = _html_to_text(response.text)
    if not title:
        title = "Unknown"
    if not content or len(content) < 100:
        raise ValueError(f"Không trích xuất được nội dung từ {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# {title}\n\n{content}",
    }


async def _crawl_with_crawl4ai(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = (result.metadata or {}).get("title", "Unknown")
        content = (result.markdown or "").strip()
        if len(content) < 100:
            raise ValueError("Crawl4AI trả về nội dung quá ngắn")
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_article(url: str) -> dict:
    try:
        return _crawl_with_requests(url)
    except Exception as req_err:
        print(f"  ⚠ requests thất bại ({req_err}) — thử Crawl4AI")
        return await _crawl_with_crawl4ai(url)


def _save_article(article: dict, filepath: Path) -> None:
    content = json.dumps(article, ensure_ascii=False, indent=2)
    if len(content.encode("utf-8")) < MIN_CONTENT_SIZE:
        raise ValueError(f"File {filepath.name} quá nhỏ sau khi crawl")
    filepath.write_text(content, encoding="utf-8")


async def crawl_all(force: bool = False) -> None:
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename

        if filepath.exists() and not force:
            size = filepath.stat().st_size
            if size >= MIN_CONTENT_SIZE:
                print(f"[{i}/{len(ARTICLE_URLS)}] ↷ Đã có: {filename}")
                continue

        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        _save_article(article, filepath)
        print(f"  ✓ Saved: {filepath.name} ({filepath.stat().st_size:,} bytes)")
