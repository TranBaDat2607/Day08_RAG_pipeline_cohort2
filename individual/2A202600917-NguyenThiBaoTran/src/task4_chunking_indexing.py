"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# --- Chunking: RecursiveCharacterTextSplitter với separator tùy biến cho VBPL 
# Lý do KHÔNG chọn MarkdownHeaderTextSplitter: MarkItDown convert .docx/.pdf
# luật ra "**Điều 1. ...**", "**Chương I**" — chữ in đậm chứ KHÔNG phải heading
# Markdown thật (#, ##), nên splitter theo header sẽ không nhận diện được ranh giới.
# Lý do KHÔNG chọn SemanticChunker: cần gọi embedding cho từng câu để dò ranh giới
# ngữ nghĩa -> chậm, tốn tài nguyên và kết quả không ổn định (non-deterministic),
# không cần thiết với văn bản pháp luật vốn đã có cấu trúc tường minh theo
# Chương/Điều/Khoản/Điểm.
# => Dùng RecursiveCharacterTextSplitter, khai báo thêm separator ưu tiên ranh
# giới "Chương" và "Điều" (xem mảng separators bên dưới). Mỗi Điều là một đơn vị
# pháp lý độc lập về ngữ nghĩa (quy định trọn vẹn một vấn đề), nên ưu tiên cắt
# tại ranh giới này giúp chunk giữ nguyên ngữ cảnh pháp lý, tránh bị cắt giữa
# chừng một quy định.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Khảo sát thực tế trong data/standardized/legal/*.md: độ dài trung bình của
# một "Điều" là ~1400-2150 ký tự (có Điều dài tới ~24.000 ký tự do chứa bảng/
# danh mục). Chọn CHUNK_SIZE=1000 để: (1) phần lớn các Điều ngắn/vừa nằm gọn
# trong 1 chunk (giữ trọn ngữ nghĩa pháp lý), (2) các Điều dài vẫn được chia
# nhỏ tại ranh giới đoạn/câu thay vì cắt ngang một từ.
# 1000 ký tự tiếng Việt cũng nằm rất an toàn trong context window của bge-m3
# (8192 token) nên không lo mất thông tin khi embed.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP=100 (10% của CHUNK_SIZE): đủ để câu/đoạn bị cắt ở ranh giới
# chunk vẫn xuất hiện trọn vẹn ở chunk liền kề — quan trọng với văn bản luật vì
# các câu thường tham chiếu chéo (vd. "quy định tại khoản 2 Điều này") và nếu
# bị cắt rời sẽ làm giảm chất lượng retrieval. Overlap quá lớn sẽ gây trùng lặp
# và lãng phí số lượng vector phải lưu/tính.
CHUNK_OVERLAP = 100

# --- Embedding: BAAI/bge-m3 -------------------------------------------------
# Lý do chọn bge-m3 thay vì all-MiniLM-L6-v2 hay OpenAI text-embedding-3-small:
#   1. Multilingual & tối ưu cho tiếng Việt: all-MiniLM-L6-v2 chủ yếu train trên
#      tiếng Anh, chất lượng embedding tiếng Việt (đặc biệt văn bản pháp luật
#      nhiều từ Hán-Việt, thuật ngữ chuyên ngành) kém hơn đáng kể so với bge-m3.
#   2. Hỗ trợ context dài (8192 token) — phù hợp với các "Điều" luật dài, hạn
#      chế phải cắt nhỏ quá mức làm vỡ ngữ cảnh.
#   3. Hỗ trợ multi-functionality (dense + sparse + ColBERT) — về sau có thể
#      tận dụng cho hybrid search trong Weaviate mà không cần đổi model.
#   4. Chạy local/open-source: dữ liệu là văn bản pháp luật/tin tức nội bộ,
#      không cần gửi qua API bên thứ ba (như OpenAI) — vừa tiết kiệm chi phí
#      vừa tránh rủi ro rò rỉ dữ liệu.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Vector store: Weaviate --------------------------------------------------
# Lý do chọn Weaviate thay vì ChromaDB hay FAISS:
#   1. Hybrid search built-in (BM25 + vector, kết hợp qua alpha): với văn bản
#      pháp luật, người dùng thường tra theo từ khóa chính xác (số Điều, tên
#      Nghị định, thuật ngữ định danh như "tiền chất", "ma túy tổng hợp") —
#      thuần dense vector search (FAISS) dễ bỏ sót các match từ khóa chính xác
#      này, trong khi BM25 lại không hiểu ngữ nghĩa. Hybrid search giải quyết
#      tốt cả hai nhu cầu.
#   2. Lưu kèm metadata (source, doc_type, chunk_index) và filter native theo
#      thuộc tính — ví dụ chỉ tìm trong "legal" hoặc chỉ trong "news".
#   3. FAISS chỉ là thư viện index thuần túy (không filter, không lưu metadata,
#      không hybrid) -> phải tự xây thêm lớp lưu trữ/metadata; ChromaDB đơn
#      giản hơn nhưng hybrid search còn hạn chế so với Weaviate.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n**Chương",  # ranh giới Chương — đơn vị lớn nhất, ưu tiên cắt trước
            "\n\n**Điều",    # ranh giới Điều — đơn vị pháp lý độc lập về ngữ nghĩa
            "\n\n",          # ranh giới đoạn / khoản
            "\n",            # ranh giới dòng / điểm (a, b, c...)
            ". ",            # ranh giới câu
            " ",
            "",
        ],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    # normalize_embeddings=True -> dùng cosine similarity khi search trong
    # vector store (Weaviate mặc định so khớp bằng cosine distance).
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    from weaviate.classes.config import Configure, DataType, Property

    from .weaviate_client import connect_weaviate

    collection_name = "DrugLawDocs"

    # connect_weaviate() dùng Weaviate local (Docker) cho dev, hoặc Weaviate
    # Cloud (qua WEAVIATE_URL/WEAVIATE_API_KEY trong .env) khi cần một cluster
    # truy cập được từ xa — vd. để backend deploy (không chạy được Docker) có
    # thể kết nối tới cùng dữ liệu đã index.
    client = connect_weaviate()
    try:
        if client.collections.exists(collection_name):
            collection = client.collections.get(collection_name)
        else:
            collection = client.collections.create(
                name=collection_name,
                # Ta tự tính embedding (bge-m3) nên tắt vectorizer mặc định của
                # Weaviate và truyền thẳng vector khi insert.
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": chunk["metadata"]["source"],
                        "doc_type": chunk["metadata"]["type"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                    },
                    vector=chunk["embedding"],
                )

        if collection.batch.failed_objects:
            print(f"  ⚠ {len(collection.batch.failed_objects)} object(s) failed to index")
    finally:
        client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
