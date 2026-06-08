"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.
"""

from pathlib import Path
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma tuý 2021 (73/2021/QH15)",
    },
    {
        "url": "https://cscnmt.khanhhoa.gov.vn/laws/detail/Nghi-dinh-Quy-dinh-chi-tiet-va-huong-dan-thi-hanh-mot-so-dieu-cua-Luat-Phong-Chong-ma-tuy-24/?download=1&id=0.pdf",
        "filename": "nghi-dinh-105-2021.pdf",
        "title": "Nghị định 105/2021/NĐ-CP hướng dẫn Luật Phòng chống ma tuý",
    },
    {
        "url": "https://jshou.edu.vn/houjs/article/download/83/90/430.pdf",
        "filename": "bo-luat-hinh-su-2015.pdf",
        "title": "Bộ luật Hình sự 2015 (quy định về tội phạm ma tuý)",
    },
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MIN_FILE_SIZE = 1024  # bytes


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str, force: bool = False) -> Path:
    """
    Tải file PDF từ URL về DATA_DIR.
    """

    filepath = DATA_DIR / filename

    if filepath.exists() and not force:
        size = filepath.stat().st_size

        if size > MIN_FILE_SIZE:
            print(f"  ↷ Đã có: {filename} ({size:,} bytes) — bỏ qua")
            return filepath

        print(f"  ⚠ File quá nhỏ ({size} bytes) — tải lại")

    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=60,
    )

    response.raise_for_status()

    content_type = response.headers.get("content-type", "")

    is_pdf = (
        "pdf" in content_type.lower()
        or response.content.startswith(b"%PDF")
    )

    if not is_pdf:
        raise ValueError(
            f"URL không trả về PDF hợp lệ:\n"
            f"{url}\n"
            f"content-type={content_type}"
        )

    if len(response.content) <= MIN_FILE_SIZE:
        raise ValueError(
            f"File tải về quá nhỏ ({len(response.content)} bytes)"
        )

    filepath.write_bytes(response.content)

    print(
        f"  ✓ Đã tải: {filename} "
        f"({len(response.content):,} bytes)"
    )

    return filepath


def download_all(force: bool = False) -> list[Path]:
    """
    Tải toàn bộ văn bản pháp luật.
    """

    setup_directory()

    downloaded = []

    for idx, doc in enumerate(LEGAL_DOCUMENTS, start=1):
        print(f"\n[{idx}/{len(LEGAL_DOCUMENTS)}] {doc['title']}")
        print(f"  URL: {doc['url']}")

        try:
            path = download_file(
                doc["url"],
                doc["filename"],
                force=force,
            )
            downloaded.append(path)

        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

    return downloaded


if __name__ == "__main__":

    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Task 1: Thu thập văn bản pháp luật")
    print("=" * 60)

    files = download_all()

    print("\n" + "=" * 60)
    print(f"✓ Hoàn tất — {len(files)} file")
    print(f"✓ Lưu tại: {DATA_DIR}")
    print("=" * 60)