"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Lưu mapping {tên file -> doc_id} để pageindex_search() dùng lại mà không
# phải submit lại tài liệu mỗi lần chạy (PageIndex xử lý tài liệu bất đồng bộ
# và mất thời gian dựng cây cấu trúc, không nên lặp lại không cần thiết).
DOC_MAP_PATH = STANDARDIZED_DIR / "_pageindex_doc_map.json"

# Cache PDF chuyển đổi từ .docx — đặt riêng (không ghi vào data/landing/) để
# giữ nguyên dữ liệu gốc, đồng thời tránh convert lại mỗi lần chạy.
PDF_CACHE_DIR = STANDARDIZED_DIR / "_pageindex_pdf_cache"

_client: PageIndexClient | None = None


def _get_client() -> PageIndexClient:
    global _client
    if _client is None:
        _client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    return _client


def _convert_docx_to_pdf(docx_path: Path) -> Path:
    """
    Convert .docx -> .pdf bằng Word COM automation (pywin32 — đã có sẵn trong
    project, không cần cài thêm gì).

    Lý do cần convert: PageIndex chỉ dựng được cây cấu trúc chính xác từ PDF
    (layout/mục lục/số trang thật — xem giải thích chi tiết ở docstring
    upload_documents). File .docx phải convert sang PDF trước khi submit.

    Kết quả được cache tại PDF_CACHE_DIR theo tên gốc — lần chạy sau bỏ qua
    nếu đã convert (Word COM khởi động chậm, không nên lặp lại không cần thiết).
    """
    pdf_path = PDF_CACHE_DIR / f"{docx_path.stem}.pdf"
    if pdf_path.exists():
        return pdf_path

    import win32com.client

    PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(str(docx_path.resolve()))
        doc.SaveAs(str(pdf_path.resolve()), FileFormat=17)  # wdFormatPDF
        doc.Close()
    finally:
        word.Quit()

    return pdf_path


def upload_documents():
    """
    Upload tài liệu lên PageIndex để PageIndex dựng "page tree" — cây cấu
    trúc phân cấp (Chương > Điều > Khoản, kèm số trang) dùng cho truy hồi
    bằng lý luận (reasoning-based retrieval) thay vì vector similarity.

    QUAN TRỌNG — vì sao upload PDF GỐC (data/landing/) thay vì markdown đã
    chuẩn hoá (data/standardized/) như mô tả gợi ý ban đầu:
    PageIndex dựng cây dựa trên CẤU TRÚC THỰC của tài liệu — với PDF là
    layout/heading/mục lục/số trang; với markdown là heading thật (#, ##...).
    Nhưng markdown ở Task 3 do MarkItDown convert lại biểu diễn "Điều X" /
    "Chương Y" bằng chữ in đậm (**...**) chứ KHÔNG phải heading Markdown thật
    (xem lý do chọn chunking strategy ở Task 4) — nếu nộp các file này,
    PageIndex sẽ không nhận diện được phân cấp Chương/Điều và dựng ra một cây
    gần như phẳng, mất hết lợi thế "structural understanding". File PDF gốc
    (vd. 73luat.pdf) giữ nguyên layout/mục lục thật, giúp PageIndex dựng cây
    chính xác hơn nhiều — đúng với triết lý "vectorless" của công cụ này.
    """
    client = _get_client()
    doc_map: dict[str, str] = {}

    pdf_files = list(LANDING_DIR.rglob("*.pdf"))

    # File .docx (vd. các nghị định/luật tải về dạng Word) không có layout PDF
    # gốc — convert sang PDF trước để PageIndex dựng cây cấu trúc được.
    for docx_path in LANDING_DIR.rglob("*.docx"):
        print(f"Converting to PDF: {docx_path.name}")
        pdf_files.append(_convert_docx_to_pdf(docx_path))

    if not pdf_files:
        print("⚠ Không tìm thấy file PDF/DOCX nào trong data/landing/")
        return

    for pdf_path in pdf_files:
        print(f"Submitting: {pdf_path.name}")
        doc_id = client.submit_document(str(pdf_path))["doc_id"]
        doc_map[pdf_path.name] = doc_id

        # PageIndex xử lý bất đồng bộ — chờ tới khi cây cấu trúc dựng xong
        # (status == "completed") thì mới có thể retrieval trên tài liệu này.
        while True:
            doc_info = client.get_document(doc_id)
            status = doc_info.get("status")
            if status == "completed":
                break
            print(f"  … đang xử lý (status={status})")
            time.sleep(5)

        print(f"  ✓ Sẵn sàng truy vấn: {pdf_path.name} (doc_id={doc_id})")

    DOC_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_MAP_PATH.write_text(json.dumps(doc_map, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ Đã lưu doc_id map tại: {DOC_MAP_PATH}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not DOC_MAP_PATH.exists():
        raise RuntimeError(
            f"Chưa có dữ liệu PageIndex — chạy upload_documents() trước "
            f"(thiếu file {DOC_MAP_PATH})"
        )

    doc_map: dict[str, str] = json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))
    client = _get_client()

    results = []
    for filename, doc_id in doc_map.items():
        # Bước 1: gửi câu hỏi — PageIndex dùng LLM "đi bộ" trên cây cấu trúc để
        # suy luận xem mục/Điều nào liên quan, thay vì so vector.
        retrieval_id = client.submit_query(doc_id=doc_id, query=query)["retrieval_id"]

        # Bước 2: poll tới khi suy luận xong
        while True:
            retrieval = client.get_retrieval(retrieval_id)
            status = retrieval.get("status")
            if status in ("completed", "failed"):
                break
            time.sleep(2)

        if status != "completed":
            continue

        # Bước 3: trải phẳng retrieved_nodes -> relevant_contents -> từng đoạn
        # nội dung liên quan. PageIndex trả các node đã sắp theo độ liên quan
        # (do LLM suy luận) nhưng KHÔNG kèm relevance score dạng số — ta suy ra
        # "score" theo thứ hạng (1/rank, cùng ý tưởng RRF ở Task 7) để giữ
        # format thống nhất, có thể so sánh/kết hợp với Task 5 & Task 6.
        rank = 0
        for node in retrieval.get("retrieved_nodes", []):
            for group in node.get("relevant_contents", []):
                for item in group:
                    rank += 1
                    results.append({
                        "content": item.get("relevant_content", ""),
                        "score": 1.0 / rank,
                        "metadata": {
                            "source": filename,
                            "node_id": node.get("node_id"),
                            "title": node.get("title"),
                            "page_index": item.get("page_index"),
                        },
                        "source": "pageindex",
                    })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
