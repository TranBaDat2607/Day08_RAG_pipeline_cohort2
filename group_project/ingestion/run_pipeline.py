"""
Chạy toàn bộ pipeline ingestion cho group project.

Nguyễn Thành Đạt (2A202600771): Data collection → chunking → FAISS index.

Usage (từ repo root):
    python -m group_project.ingestion.run_pipeline
    python -m group_project.ingestion.run_pipeline --skip-collect --skip-crawl
    python -m group_project.ingestion.run_pipeline --copy-from-individual
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INDIVIDUAL_DATA = (
    _REPO_ROOT
    / "individual"
    / "2A202600771-NguyenThanhDat"
    / "data"
)

from .chunk_and_index import faiss_search, run_pipeline as run_chunk_index
from .collect_legal import download_all
from .convert_markdown import convert_all
from .crawl_news import crawl_all
from .config import DATA_DIR, FAISS_INDEX_FILE, LANDING_DIR, STANDARDIZED_DIR


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def copy_from_individual() -> None:
    """Copy dữ liệu đã có từ bài cá nhân sang group_project/data/."""
    if not _INDIVIDUAL_DATA.exists():
        raise FileNotFoundError(f"Không tìm thấy: {_INDIVIDUAL_DATA}")

    for sub in ("landing", "standardized"):
        src = _INDIVIDUAL_DATA / sub
        dst = DATA_DIR / sub
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"✓ Copied {sub}/ → {dst}")


def run_all(
    skip_collect: bool = False,
    skip_crawl: bool = False,
    skip_convert: bool = False,
    force: bool = False,
) -> None:
    _configure_stdout()

    print("=" * 60)
    print("Group Project — Ingestion Pipeline (FAISS)")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 60)

    if not skip_collect:
        print("\n[1/4] Thu thập văn bản pháp luật (Task 1)")
        download_all(force=force)
    else:
        print("\n[1/4] Bỏ qua Task 1 (skip-collect)")

    if not skip_crawl:
        print("\n[2/4] Crawl bài báo (Task 2)")
        asyncio.run(crawl_all(force=force))
    else:
        print("\n[2/4] Bỏ qua Task 2 (skip-crawl)")

    if not skip_convert:
        print("\n[3/4] Convert Markdown (Task 3)")
        convert_all()
    else:
        print("\n[3/4] Bỏ qua Task 3 (skip-convert)")

    print("\n[4/4] Chunking + FAISS Index (Task 4)")
    run_chunk_index()

    print("\n" + "=" * 60)
    print("✓ Hoàn tất ingestion pipeline")
    print(f"  Landing     : {LANDING_DIR}")
    print(f"  Standardized: {STANDARDIZED_DIR}")
    print(f"  FAISS index : {FAISS_INDEX_FILE}")
    print("=" * 60)

    if FAISS_INDEX_FILE.exists():
        print("\n--- Demo FAISS search ---")
        for r in faiss_search("hình phạt tàng trữ ma tuý", top_k=3):
            src = r["metadata"].get("source", "?")
            print(f"  [{r['score']:.3f}] {src}: {r['content'][:80]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Group project ingestion pipeline")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--force", action="store_true", help="Tải/crawl lại dù đã có file")
    parser.add_argument(
        "--copy-from-individual",
        action="store_true",
        help="Copy data từ individual/2A202600771-NguyenThanhDat/data rồi chỉ index FAISS",
    )
    args = parser.parse_args()

    if args.copy_from_individual:
        _configure_stdout()
        copy_from_individual()
        run_chunk_index()
        if FAISS_INDEX_FILE.exists():
            print("\n--- Demo FAISS search ---")
            for r in faiss_search("hình phạt tàng trữ ma tuý", top_k=3):
                src = r["metadata"].get("source", "?")
                print(f"  [{r['score']:.3f}] {src}: {r['content'][:80]}...")
        return

    run_all(
        skip_collect=args.skip_collect,
        skip_crawl=args.skip_crawl,
        skip_convert=args.skip_convert,
        force=args.force,
    )


if __name__ == "__main__":
    main()
