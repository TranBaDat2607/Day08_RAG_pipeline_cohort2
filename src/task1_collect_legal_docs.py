"""
Task 1 - Thu thap van ban phap luat ve ma tuy va cac chat cam (tu dong tu vbpl.vn).

Crawl Co so du lieu quoc gia ve van ban phap luat (https://vbpl.vn/ - React SPA)
bang Playwright. Voi moi tu khoa trong SEARCH_KEYWORDS:
    1. Dat pham vi tim kiem = ALL (toan quoc) truoc moi lan tim.
    2. Nhap tu khoa vao o tim kiem va bam "Tim kiem".
    3. Duyet qua tat ca cac trang ket qua.
    4. Chi giu van ban co trang thai "Con hieu luc" (tag mau xanh rgb(0,181,77))
       VA thuoc loai van ban cho phep (Luat, Bo luat, Nghi dinh, Thong tu,
       Thong tu lien tich).
    5. Tai file PDF goc cua cac van ban do ve data/landing/legal/.
    6. Loai bo file trung lap (theo ID van ban khi crawl + theo noi dung SHA-256).

Co che tai PDF (phat hien qua khao sat site):
    - Bam nut "PDF" tren the ket qua -> mo tab chi tiet .../chi-tiet/{slug}--{ID}
    - Trinh xem PDF tai file goc tu gateway MinIO:
        https://vbpl-bientap-gateway.moj.gov.vn/.../buckets/vbpl/{ID}/{file}.pdf/download
    - Ta bat URL nay tu network roi tai ve bang requests (endpoint public, khong can auth).

Chay:
    python -m src.task1_collect_legal_docs                # tai toi da MAX_FILES file
    python -m src.task1_collect_legal_docs --max 50       # doi cap so file
    python -m src.task1_collect_legal_docs --headed       # hien trinh duyet de debug

Yeu cau: pip install playwright requests ; python -m playwright install chromium
"""

import argparse
import hashlib
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import requests
from playwright.sync_api import sync_playwright

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tao thu muc data/landing/legal/ neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Thu muc da san sang: {DATA_DIR}")


# Tu khoa giu nguyen dau tieng Viet vi day la noi dung go vao o tim kiem cua site
SEARCH_KEYWORDS = ["ma túy", "chất ma túy", "chất gây nghiện", "chất hướng thần", "tiền chất"]

# =============================================================================
# CAU HINH - chinh o day
# =============================================================================

BASE_URL          = "https://vbpl.vn/"
VALID_STATUS_TEXT = "Con hieu luc"
VALID_COLOR_RGB   = "rgb(0,181,77)"          # da chuan hoa bo khoang trang
# Chi giu cac loai van ban nay (so khop theo phan dau tieu de, khong dau)
ALLOWED_DOC_TYPES = ["Bo luat", "Luat", "Nghi dinh", "Thong tu lien tich", "Thong tu"]

MAX_FILES         = 30      # CAP: tong so file PDF tai ve (None = khong gioi han)
MAX_PER_KEYWORD   = None    # tran an toan so file moi tu khoa (None = khong gioi han)
MAX_PAGES         = 15      # tran an toan so trang phan trang moi tu khoa

HEADLESS          = True
NAV_TIMEOUT_MS    = 30_000
RESULTS_TIMEOUT_MS = 40_000
PDF_WAIT_S        = 18      # thoi gian cho tab PDF nap URL file goc
POLITE_DELAY_S    = 1.5     # nghi giua cac luot tai cho lich su voi server
DOWNLOAD_RETRIES  = 3
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Selectors (xac nhan qua khao sat truc tiep site)
SEL_KEYWORD     = "input#keyword"
# Nut tim kiem chon theo cau truc (nut primary canh o keyword) de khong phu thuoc chu Viet
SEL_SEARCH_BTN  = "span.ant-input-affix-wrapper:has(#keyword) button.ant-btn-primary"
SEL_SCOPE_ALL   = "input.ant-radio-input[value='ALL'][name='scopeArea']"
SEL_CARD        = ".DocumentCard_cardContent__5tVRC"
SEL_CARD_TITLE  = ".DocumentCard_documentTitle__aE_F_"
SEL_CARD_STATUS = "span.text-xs.white-space-nowrap"
SEL_NEXT_PAGE   = "li.ant-pagination-next"

# Bo bat URL PDF goc do trinh duyet tai (dien boi response listener)
_captured_pdf_urls: list[str] = []


# =============================================================================
# HAM TIEN ICH
# =============================================================================

def _strip_diacritics(s: str) -> str:
    """Bo dau tieng Viet, doi d/D, tra ve chu thuong."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def slugify_vietnamese(name: str, max_len: int = 90) -> str:
    """Chuyen ten thanh slug khong dau, an toan cho ten file Windows."""
    s = _strip_diacritics(unquote(name))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len].strip("-") or "van-ban"


def is_valid_status(text: str | None, color: str | None) -> bool:
    """True neu trang thai la 'Con hieu luc' (theo text hoac mau rgb(0,181,77))."""
    if text and _strip_diacritics(text.strip()) == _strip_diacritics(VALID_STATUS_TEXT):
        return True
    if color and color.replace(" ", "") == VALID_COLOR_RGB:
        return True
    return False


_ALLOWED_NORMS = [(_strip_diacritics(t), t) for t in ALLOWED_DOC_TYPES]


def matched_doc_type(title: str) -> str | None:
    """Tra ve loai van ban neu tieu de bat dau bang mot loai cho phep, nguoc lai None."""
    norm = _strip_diacritics(title).lstrip()
    for ntype, original in _ALLOWED_NORMS:
        if norm.startswith(ntype):
            return original
    return None


def extract_doc_id(detail_url: str) -> str:
    """Lay ID van ban (phan sau '--') tu URL trang chi tiet, dung lam khoa khu trung."""
    path = detail_url.split("?")[0]
    return path.split("--")[-1] if "--" in path else path.rstrip("/").split("/")[-1]


def _is_pdf_response(url: str) -> bool:
    u = url.lower()
    return ".pdf" in u and (
        "vbpl-bientap-gateway" in u or "/buckets/vbpl/" in u or "filedata" in u
    )


def download_pdf(url: str, dest: Path) -> bool:
    """Tai PDF tu URL gateway ve dest bang requests. Tra ve True neu hop le."""
    headers = {"User-Agent": USER_AGENT, "Referer": BASE_URL}
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 1024:
                dest.write_bytes(r.content)
                return True
            print(f"    [WARN] Lan {attempt}: status={r.status_code}, "
                  f"size={len(r.content)}, head={r.content[:5]!r}")
        except Exception as e:
            print(f"    [WARN] Lan {attempt} loi tai: {e}")
        time.sleep(1.5 * attempt)
    return False


# =============================================================================
# THAO TAC TREN TRANG
# =============================================================================

def set_scope_all(page):
    """Dam bao pham vi tim kiem = ALL (toan quoc) truoc moi lan tim."""
    radio = page.locator(SEL_SCOPE_ALL)
    try:
        if radio.count() and not radio.first.is_checked():
            # AntD an input goc -> click vao wrapper cha
            radio.first.locator("xpath=ancestor::label[contains(@class,'ant-radio-wrapper')]").click()
        # xac nhan
        ok = radio.count() and radio.first.is_checked()
        print(f"    Pham vi = ALL ({'da chon' if ok else 'KHONG xac nhan duoc'})")
    except Exception as e:
        print(f"    [WARN] Khong dat duoc pham vi ALL: {e}")


def wait_for_results(page):
    """Cho toi khi the ket qua that (khong con skeleton) hien thi."""
    page.wait_for_function(
        """() => {
            const cs = document.querySelectorAll('.DocumentCard_cardContent__5tVRC');
            if (!cs.length) return false;
            return [...cs].some(c => !c.querySelector('.ant-skeleton')
                                     && c.textContent.trim().length > 40);
        }""",
        timeout=RESULTS_TIMEOUT_MS,
    )
    page.wait_for_timeout(1200)


def do_search(page, keyword: str):
    """Dat pham vi ALL, nhap tu khoa va bam Tim kiem."""
    set_scope_all(page)
    kw = page.locator(SEL_KEYWORD).first
    kw.fill("")
    kw.fill(keyword)
    page.locator(SEL_SEARCH_BTN).first.click()
    wait_for_results(page)


def first_card_title(page) -> str:
    cards = page.locator(SEL_CARD)
    if cards.count() == 0:
        return ""
    try:
        return cards.nth(0).locator(SEL_CARD_TITLE).inner_text().strip()
    except Exception:
        return ""


def card_status(card):
    """Doc (text, color) cua tag trang thai trong mot the ket qua."""
    spans = card.locator(SEL_CARD_STATUS)
    for j in range(spans.count()):
        try:
            t = spans.nth(j).inner_text().strip()
        except Exception:
            continue
        if "hieu luc" in _strip_diacritics(t):
            try:
                color = spans.nth(j).evaluate("e => getComputedStyle(e).color")
            except Exception:
                color = None
            return t, color
    return None, None


def go_next_page(page) -> bool:
    """Chuyen sang trang ket qua ke tiep. Tra ve False neu da het trang."""
    nxt = page.locator(SEL_NEXT_PAGE)
    if nxt.count() == 0:
        return False
    cls = nxt.first.get_attribute("class") or ""
    if "disabled" in cls:
        return False
    before = first_card_title(page)
    try:
        nxt.first.locator("button").click()
    except Exception:
        return False
    # cho danh sach doi (tieu de the dau khac di)
    try:
        page.wait_for_function(
            """(prev) => {
                const c = document.querySelector('.DocumentCard_documentTitle__aE_F_');
                return c && c.textContent.trim().length > 0 && c.textContent.trim() !== prev;
            }""",
            arg=before, timeout=RESULTS_TIMEOUT_MS,
        )
        page.wait_for_timeout(1000)
        return True
    except Exception:
        return False


def fetch_pdf_url_for_card(ctx, card) -> tuple[str | None, str | None]:
    """
    Bam nut PDF cua the -> mo tab chi tiet -> bat URL file PDF goc.
    Tra ve (doc_id, pdf_url) hoac (None, None) neu that bai.
    """
    start = len(_captured_pdf_urls)
    try:
        with ctx.expect_page(timeout=12_000) as pop:
            card.get_by_role("button", name=re.compile("^PDF$")).first.click()
        detail = pop.value
    except Exception as e:
        print(f"    [WARN] Khong mo duoc tab PDF: {e}")
        return None, None

    doc_id = None
    pdf_url = None
    try:
        detail.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT_MS)
        doc_id = extract_doc_id(detail.url)
        # cho trinh xem nap URL file goc
        deadline = time.time() + PDF_WAIT_S
        while time.time() < deadline:
            for u in _captured_pdf_urls[start:]:
                if doc_id and f"/{doc_id}/" in u:
                    pdf_url = u
                    break
            if pdf_url is None and len(_captured_pdf_urls) > start:
                pdf_url = _captured_pdf_urls[-1]  # fallback: URL pdf moi nhat
            if pdf_url:
                break
            detail.wait_for_timeout(500)
    except Exception as e:
        print(f"    [WARN] Loi khi doc tab chi tiet: {e}")
    finally:
        try:
            detail.close()
        except Exception:
            pass
    return doc_id, pdf_url


# =============================================================================
# KHU TRUNG LAP THEO NOI DUNG
# =============================================================================

def dedupe_by_hash(dest_dir: Path) -> int:
    """Xoa cac file trung noi dung (SHA-256), giu lai ten ngan nhat. Tra ve so file da xoa."""
    by_hash: dict[str, list[Path]] = {}
    for f in dest_dir.glob("*.pdf"):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        by_hash.setdefault(h, []).append(f)
    removed = 0
    for files in by_hash.values():
        if len(files) > 1:
            files.sort(key=lambda p: (len(p.name), p.name))
            for dup in files[1:]:
                dup.unlink()
                removed += 1
    if removed:
        print(f"[OK] Da loai {removed} file trung noi dung")
    return removed


# =============================================================================
# CRAWL
# =============================================================================

def crawl_keyword(ctx, page, keyword, seen_doc_ids, saved) -> int:
    """Crawl 1 tu khoa; tai PDF cac van ban 'Con hieu luc' thuoc loai cho phep.
    Tra ve so file da tai cho tu khoa nay. `saved` la list dung dem tong (mutable)."""
    print(f"\n{'='*60}\nTu khoa: {keyword!r}\n{'='*60}")
    try:
        do_search(page, keyword)
    except Exception as e:
        print(f"  [ERR] Tim kiem that bai: {e}")
        return 0

    kw_count = 0
    for page_no in range(1, MAX_PAGES + 1):
        cards = page.locator(SEL_CARD)
        n = cards.count()
        print(f"  - Trang {page_no}: {n} ket qua")

        for i in range(n):
            if MAX_FILES is not None and len(saved) >= MAX_FILES:
                print("  [STOP] Dat MAX_FILES, dung.")
                return kw_count
            if MAX_PER_KEYWORD is not None and kw_count >= MAX_PER_KEYWORD:
                print("  [STOP] Dat MAX_PER_KEYWORD, sang tu khoa khac.")
                return kw_count

            card = cards.nth(i)
            try:
                title = card.locator(SEL_CARD_TITLE).inner_text().strip()
            except Exception:
                continue
            text, color = card_status(card)

            if not is_valid_status(text, color):
                continue
            dtype = matched_doc_type(title)
            if dtype is None:
                continue

            print(f"    - [{dtype}] {title[:70]}")
            doc_id, pdf_url = fetch_pdf_url_for_card(ctx, card)
            if doc_id and doc_id in seen_doc_ids:
                print("      Bo qua (trung ID da tai)")
                continue
            if not pdf_url:
                print("      [WARN] Khong tim thay file PDF goc (co the chi co DOCX) - bo qua")
                continue

            fname = f"{slugify_vietnamese(title)}--{doc_id}.pdf"
            dest = DATA_DIR / fname
            if download_pdf(pdf_url, dest):
                if doc_id:
                    seen_doc_ids.add(doc_id)
                saved.append(dest)
                kw_count += 1
                print(f"      [OK] Da tai [{len(saved)}]: {fname}")
            else:
                print(f"      [ERR] Tai that bai: {pdf_url}")
            time.sleep(POLITE_DELAY_S)

        if MAX_FILES is not None and len(saved) >= MAX_FILES:
            return kw_count
        if not go_next_page(page):
            print("  [OK] Het trang ket qua.")
            break
    return kw_count


def collect_all_legal_docs(max_files=None, headless=None):
    """Orchestrator: crawl tat ca tu khoa, khu trung, in tong ket."""
    global MAX_FILES, HEADLESS
    if max_files is not None:
        MAX_FILES = max_files
    if headless is not None:
        HEADLESS = headless

    setup_directory()
    print(f"Cau hinh: MAX_FILES={MAX_FILES}, headless={HEADLESS}, "
          f"loai VB={ALLOWED_DOC_TYPES}")

    seen_doc_ids: set[str] = set()
    saved: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        ctx = browser.new_context(user_agent=USER_AGENT, accept_downloads=True)
        ctx.on("response", lambda r: _captured_pdf_urls.append(r.url)
               if _is_pdf_response(r.url) else None)
        page = ctx.new_page()
        page.set_default_timeout(NAV_TIMEOUT_MS)
        page.set_default_navigation_timeout(NAV_TIMEOUT_MS)
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(3500)

        try:
            for kw in SEARCH_KEYWORDS:
                if MAX_FILES is not None and len(saved) >= MAX_FILES:
                    break
                crawl_keyword(ctx, page, kw, seen_doc_ids, saved)
        finally:
            browser.close()

    print(f"\n{'='*60}")
    print(f"[OK] Tai xong {len(saved)} file PDF (van ban 'Con hieu luc', loai cho phep)")
    dedupe_by_hash(DATA_DIR)
    total = len(list(DATA_DIR.glob('*.pdf')))
    print(f"[OK] Hoan tat. Tong so PDF trong {DATA_DIR.name}/: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl van ban phap luat 'Con hieu luc' tu vbpl.vn")
    parser.add_argument("--max", type=int, default=None, help="So file PDF toi da (override MAX_FILES)")
    parser.add_argument("--headed", action="store_true", help="Hien trinh duyet (debug)")
    args = parser.parse_args()
    collect_all_legal_docs(
        max_files=args.max,
        headless=(False if args.headed else None),
    )
