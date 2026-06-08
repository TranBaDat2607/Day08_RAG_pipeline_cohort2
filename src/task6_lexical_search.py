"""
Task 6 — Lexical Search Module (BM25 mặc định + các biến thể bonus).

Mặc định dùng BM25 (rank-bm25). Phần BONUS (+5đ) ở cuối file implement/giải thích
3 cơ chế lexical search KHÁC: TF-IDF (sklearn), Elasticsearch, và Weaviate BM25
built-in — kèm giải thích cơ chế bằng tiếng Việt.

Cài đặt:
    pip install rank-bm25          # BM25 mặc định
    pip install scikit-learn       # cho biến thể TF-IDF (bonus)
    pip install underthesea        # (tuỳ chọn) word-segmentation tiếng Việt

Vì sao lexical search vẫn cần thiết khi đã có semantic search (Task 5)?
    - Dense/semantic search giỏi "hiểu nghĩa" nhưng HAY TRƯỢT ở các token chính
      xác, hiếm gặp: số điều luật ("Điều 248"), số nghị định ("105/2021/NĐ-CP"),
      tên riêng nghệ sĩ. Đây đúng là điểm mạnh của lexical (khớp đúng mặt chữ).
    - Vì thế Task 9 gộp dense (Task 5) + lexical (Task 6) bằng RRF → bù khuyết
      điểm cho nhau (hybrid search).

BM25 (Best Matching 25) hoạt động thế nào — công thức:
    score(q, d) = Σ_{t ∈ q} IDF(t) · ( tf(t,d)·(k1+1) ) / ( tf(t,d) + k1·(1 - b + b·|d|/avgdl) )
    Trong đó:
    - tf(t,d)  : số lần token t xuất hiện trong document d (Term Frequency).
    - IDF(t)   : Inverse Document Frequency — token hiếm (ít doc chứa) → trọng số
                 cao; token phổ biến ("của", "và") → trọng số gần 0.
    - |d|/avgdl: độ dài doc chia độ dài trung bình — chuẩn hoá độ dài (length
                 normalization) để doc dài không tự động thắng chỉ vì nhiều chữ.
    - k1 = 1.5 : kiểm soát "term saturation" — lặp 1 từ 10 lần KHÔNG tốt gấp 10
                 lần lặp 1 lần (lợi ích bão hoà dần).
    - b  = 0.75: mức độ áp dụng length normalization (0 = tắt, 1 = tối đa).
    Khác TF-IDF cổ điển ở chỗ: BM25 có (a) bão hoà tf và (b) chuẩn hoá độ dài —
    nên BM25 ổn định hơn TF-IDF trên corpus có document dài ngắn lệch nhau.
"""

from functools import lru_cache

from src.task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    normalize_text,
)


# =============================================================================
# Tokenization tiếng Việt
# =============================================================================
# Lexical search KHÁC dense ở chỗ phải KHỚP ĐÚNG token → tokenize tốt rất quan
# trọng. Tiếng Việt: 1 "từ" có thể gồm nhiều "tiếng" ("ma tuý", "tàng trữ").
# Nếu có underthesea: word_tokenize gộp "ma tuý" -> "ma_tuý" (1 token) giúp BM25
# phân biệt cụm từ chính xác hơn. Không có thì fallback regex tách theo tiếng.
# QUAN TRỌNG: query và corpus PHẢI dùng CÙNG hàm tokenize, nếu không token sẽ
# không bao giờ khớp (vd corpus "ma_tuý" còn query ["ma","tuý"] -> score = 0).

import re

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Trả về hàm tokenize. Ưu tiên underthesea, fallback regex tách \\w+."""
    try:
        from underthesea import word_tokenize

        def tokenize(text: str) -> list[str]:
            # format="text" -> "ma tuý" thành "ma_tuý"; rồi tách theo space.
            segmented = word_tokenize(text.lower(), format="text")
            return segmented.split()

        return tokenize
    except Exception:
        # Fallback: tách theo ký tự chữ-số (\w+), giữ được dấu tiếng Việt (UNICODE).
        def tokenize(text: str) -> list[str]:
            return _WORD_RE.findall(text.lower())

        return tokenize


def _tokenize(text: str) -> list[str]:
    return _get_tokenizer()(text)


# =============================================================================
# Nạp corpus
# =============================================================================
# Đọc corpus TỪ CHÍNH ChromaDB mà Task 4 đã index — để BM25 chạy trên ĐÚNG bộ
# chunk giống hệt semantic search (Task 5). Nhờ vậy khi Task 9 gộp dense + lexical
# bằng RRF, hai bên xếp hạng trên cùng một không gian chunk → merge nhất quán.
# Nếu Chroma trống/chưa có (chưa chạy Task 4), fallback chunk trực tiếp từ
# data/standardized/ để module vẫn chạy độc lập được.

def _load_corpus() -> list[dict]:
    """Trả về list of {'content': str, 'metadata': dict}."""
    # 1) Ưu tiên đọc từ Chroma collection (đồng bộ với Task 5).
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        data = collection.get(include=["documents", "metadatas"])
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        if documents:
            return [
                {"content": doc, "metadata": dict(meta or {})}
                for doc, meta in zip(documents, metadatas)
            ]
    except Exception:
        pass

    # 2) Fallback: chunk lại từ markdown standardized (dùng đúng logic Task 4).
    try:
        from src.task4_chunking_indexing import load_documents, chunk_documents

        return chunk_documents(load_documents())
    except Exception:
        return []


# =============================================================================
# BM25 index (mặc định)
# =============================================================================

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi đã fit trên corpus đã tokenize.
    """
    from rank_bm25 import BM25Okapi

    # Tokenize từng document bằng tokenizer tiếng Việt (xem giải thích ở trên).
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    # k1=1.5, b=0.75 là mặc định "kinh điển" của BM25 — cân bằng tốt cho corpus
    # văn bản luật + báo (độ dài chunk khá đồng đều quanh CHUNK_SIZE=1000).
    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


@lru_cache(maxsize=1)
def _get_index():
    """Load corpus + build BM25 index 1 lần rồi cache (tránh dựng lại mỗi query)."""
    corpus = _load_corpus()
    if not corpus:
        return None, []
    bm25 = build_bm25_index(corpus)
    return bm25, corpus


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score (càng cao càng khớp)
            'metadata': dict
        }
        Sorted by score descending, độ dài ≤ top_k.
    """
    # Chuẩn hoá query giống lúc index (NFC + gọn whitespace) cho nhất quán.
    query = normalize_text(query)
    if not query:
        return []

    bm25, corpus = _get_index()
    if bm25 is None or not corpus:
        return []

    # Tokenize query bằng CÙNG tokenizer với corpus (bắt buộc để token khớp).
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    # BM25 chấm điểm query với TỪNG document trong corpus.
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k index có điểm cao nhất (sort giảm dần). Chỉ giữ score > 0 —
    # score = 0 nghĩa là không có token nào của query khớp document đó.
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": dict(corpus[idx]["metadata"]),
        })
    return results


# =============================================================================
# BONUS (+5đ) — Các phương pháp lexical search KHÁC BM25 (giải thích cơ chế)
# =============================================================================

# -----------------------------------------------------------------------------
# BIẾN THỂ 1: TF-IDF + Cosine similarity (scikit-learn)
# -----------------------------------------------------------------------------
# Cơ chế:
#   - Biểu diễn mỗi document thành 1 vector thưa (sparse) trên không gian từ vựng:
#     mỗi chiều = 1 token, giá trị = TF-IDF = tf(t,d) · idf(t).
#         tf  : tần suất token trong document.
#         idf : log( N / df(t) ) — token xuất hiện ở càng ít document càng "đặc
#               trưng" → trọng số cao; "của", "và" gần như bằng 0.
#   - Query cũng được vector hoá y hệt, rồi xếp hạng document theo COSINE
#     SIMILARITY giữa vector query và vector document.
# Khác BM25 ở đâu?
#   - TF-IDF KHÔNG bão hoà tf (lặp 10 lần ~ điểm gấp 10) và chuẩn hoá độ dài chỉ
#     gián tiếp qua cosine (chuẩn L2), trong khi BM25 chuẩn hoá theo avgdl tường
#     minh → BM25 thường nhỉnh hơn trên corpus document dài-ngắn lệch nhau.
#   - Bù lại TF-IDF rất nhanh, có sẵn trong sklearn, dễ giải thích.

@lru_cache(maxsize=1)
def _get_tfidf_index():
    """Build TF-IDF matrix 1 lần và cache (vectorizer + ma trận document)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = _load_corpus()
    if not corpus:
        return None, None, []

    # Dùng cùng tokenizer tiếng Việt; tắt lowercase của sklearn vì _tokenize đã
    # lower (tránh xử lý 2 lần). token_pattern=None để sklearn dùng tokenizer ta.
    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, lowercase=False)
    matrix = vectorizer.fit_transform([d["content"] for d in corpus])
    return vectorizer, matrix, corpus


def tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """
    [BONUS] Lexical search bằng TF-IDF + cosine similarity (thay cho BM25).
    Cùng contract trả về với lexical_search().
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query = normalize_text(query)
    if not query:
        return []

    vectorizer, matrix, corpus = _get_tfidf_index()
    if vectorizer is None or not corpus:
        return []

    query_vec = vectorizer.transform([query])
    # cosine giữa query và mọi document → mảng điểm 1 chiều.
    scores = cosine_similarity(query_vec, matrix)[0]

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": dict(corpus[idx]["metadata"]),
        })
    return results


# -----------------------------------------------------------------------------
# BIẾN THỂ 2: Elasticsearch (BM25 phân tán, production-grade)
# -----------------------------------------------------------------------------
# Cơ chế:
#   - Elasticsearch (dựa trên Apache Lucene) dùng INVERTED INDEX: ánh xạ
#     token -> danh sách document chứa token đó, kèm vị trí & tần suất. Nhờ vậy
#     truy vấn không cần quét toàn corpus (như rank-bm25) mà tra index O(số token
#     trong query) → mở rộng tới hàng triệu document.
#   - Mặc định ES chấm điểm bằng CHÍNH BM25 (cùng công thức k1/b ở trên), cộng
#     thêm analyzer (tách từ, hạ chữ, bỏ dấu/stopword) cấu hình được cho tiếng
#     Việt. Là lựa chọn khi cần scale + lọc metadata + highlight ở production.
#   - Điểm số ES trả về cũng là BM25 score → giữ nguyên contract dict bên dưới.
#
# def elasticsearch_search(query: str, top_k: int = 10) -> list[dict]:
#     from elasticsearch import Elasticsearch
#     es = Elasticsearch("http://localhost:9200")
#     resp = es.search(
#         index="drug_law_docs",
#         query={"match": {"content": normalize_text(query)}},  # match -> BM25
#         size=top_k,
#     )
#     return [
#         {
#             "content": hit["_source"]["content"],
#             "score": float(hit["_score"]),          # _score chính là BM25
#             "metadata": hit["_source"].get("metadata", {}),
#         }
#         for hit in resp["hits"]["hits"]
#     ]


# -----------------------------------------------------------------------------
# BIẾN THỂ 3: Weaviate BM25 built-in (cùng store với dense → hybrid 1 call)
# -----------------------------------------------------------------------------
# Cơ chế:
#   - Weaviate cũng cài BM25f (biến thể BM25 cho nhiều field) ngay trong vector
#     DB. Lợi thế lớn: DỮ LIỆU NẰM CHUNG với vector dense → có thể gọi
#     collection.query.hybrid(alpha=...) để Weaviate tự gộp BM25 + dense bằng
#     fusion (RRF/relative score) trong MỘT request, không phải merge thủ công
#     như Task 9 đang làm với Chroma + rank-bm25.
#   - alpha = 0  -> thuần BM25 (lexical); alpha = 1 -> thuần vector; 0<alpha<1 ->
#     hybrid. Đây chính là lý do README khuyến nghị Weaviate cho hybrid search.
#
# def weaviate_bm25_search(query: str, top_k: int = 10) -> list[dict]:
#     import weaviate
#     client = weaviate.connect_to_local()              # hoặc connect_to_weaviate_cloud
#     collection = client.collections.get("DrugLawDocs")
#     resp = collection.query.bm25(                     # BM25 built-in (lexical thuần)
#         query=normalize_text(query),
#         limit=top_k,
#         return_metadata=["score"],
#     )
#     return [
#         {
#             "content": o.properties["content"],
#             "score": float(o.metadata.score),
#             "metadata": o.properties.get("metadata", {}),
#         }
#         for o in resp.objects
#     ]


if __name__ == "__main__":
    # Test nhanh BM25 (mặc định)
    print("=== BM25 (default) ===")
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

    # Test biến thể bonus TF-IDF (nếu đã cài scikit-learn)
    print("\n=== TF-IDF (bonus) ===")
    try:
        for r in tfidf_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5):
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
    except ImportError:
        print("scikit-learn chưa cài — bỏ qua demo TF-IDF.")
