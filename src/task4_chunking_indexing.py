"""
Task 4 - Chunking & Indexing vao Vector Store.

Pipeline: load_documents -> chunk_documents -> embed_chunks -> index_to_vectorstore

Tong ket cac lua chon (giai thich chi tiet tai tung CONFIG ben duoi):
    - Chunking : RecursiveCharacterTextSplitter voi separator nhan biet cau truc
                 van ban luat tieng Viet ("Chuong", "Dieu").
    - Embedding: intfloat/multilingual-e5-small (384 dim, ~470MB) - ban NHE, chay
                 nhanh tren CPU, van da ngon ngu & tot cho tieng Viet.
                 (BAAI/bge-m3 1024 dim la ban nang/chat luong cao hon neu co GPU.)
    - Store    : ChromaDB (local, persistent) - khong can Docker/server, chay
                 ngay tren Windows. (Weaviate la alternative neu co Docker.)

Cai dat:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import unicodedata
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "DrugLawDocs"


# =============================================================================
# CONFIGURATION - Giai thich lua chon
# =============================================================================

# --- Chunking ---------------------------------------------------------------
# CHUNK_SIZE = 1000 ky tu: mot "Dieu" (article) hoac vai "Khoan" (clause) trong
#   van ban luat VN thuong dai ~500-1500 ky tu. 1000 du de giu tron mot don vi
#   ngu nghia phap ly ma van nho de retrieval chinh xac (~300-400 token).
# CHUNK_OVERLAP = 150 (~15%): giu ngu canh khi mot Khoan bi cat sang chunk ke,
#   tranh mat nghia o ranh gioi chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"
# Vi sao KHONG dung markdown_header? MarkItDown convert PDF luat ra plain text -
# "Dieu 1", "Chuong I" la dong text thuong, KHONG phai heading "#". Nen ta dung
# recursive splitter va tu them separator "\nChuong "/"\nDieu " de uu tien cat
# dung ranh gioi dieu luat (cau truc tu nhien cua van ban phap luat VN).

# --- Embedding --------------------------------------------------------------
# intfloat/multilingual-e5-small: ban NHE (~470MB, 384 dim) - embed nhanh tren
#   CPU, da ngon ngu, tieng Viet tot. Dung tokenizer subword rieng -> KHONG
#   word-segment (khong noi "tri_tue_nhan_tao"); underscore la out-of-distribution
#   va lam GIAM chat luong dense retrieval. Chi chuan hoa NFC + whitespace.
# Muon chat luong cao nhat (co GPU): doi sang "BAAI/bge-m3" + EMBEDDING_DIM=1024
#   va BO prefix query:/passage: ben duoi (bge-m3 khong dung prefix).
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# Quy uoc prefix cua ho model E5 (BAT BUOC de retrieval dung):
#   - document/chunk khi index  -> "passage: " + text
#   - query khi search (Task 5) -> "query: "   + text
# Task 5 import EMBED_QUERY_PREFIX de giu nhat quan voi index o day.
EMBED_PASSAGE_PREFIX = "passage: "
EMBED_QUERY_PREFIX = "query: "

# --- Vector store -----------------------------------------------------------
# ChromaDB: local persistent, khong can Docker/server, cosine similarity.
#   (Weaviate can Docker - khong kha dung o moi truong hien tai tren Windows.)
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# Chuan hoa tieng Viet (KHONG word-segmentation - xem giai thich o EMBEDDING)
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Chuan hoa an toan, dung chung cho moi text:
        - NFC: hop nhat 2 cach ma hoa dau tieng Viet (to hop vs dung san) -> tranh
          loi "cung chu nhung khac bytes" khien embedding/BM25 khong khop.
        - Gon whitespace: bo space thua, gop >2 dong trong thanh 1 dong trong.
    KHONG dung toi dau/chu hoa-thuong (model tu xu ly), KHONG tach tu.
    """
    text = unicodedata.normalize("NFC", text)
    # Bo khoang trang cuoi dong, gop nhieu dong trong lien tiep
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Gop space/tab lap trong cung dong (OCR hay tao space thua)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Doc toan bo markdown files tu data/standardized/ va chuan hoa NFC.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        content = normalize_text(content)
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bang RecursiveCharacterTextSplitter.

    Separator duoc sap theo do uu tien de CAT DUNG ranh gioi van ban luat VN:
        "\\nChuong " > "\\nDieu " > doan > dong > cau > tu > ky tu.
    Nho vay moi chunk co xu huong la mot Dieu/Khoan tron ven khi vua kich thuoc.

    Returns:
        List of {'content': str, 'metadata': dict} - moi item la 1 chunk.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # keep_separator=True (mac dinh) -> tien to "Dieu ..." duoc giu o dau chunk
        separators=["\nChương ", "\nĐiều ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bang intfloat/multilingual-e5-small.

    - Them prefix "passage: " cho moi chunk (quy uoc E5) - chi o buoc embed,
      KHONG dung toi content goc luu trong store.
    - normalize_embeddings=True -> vector don vi, cosine = dot product (khop voi
      khong gian "cosine" cua Chroma).

    Returns:
        Moi chunk dict duoc them key 'embedding': list[float].
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [EMBED_PASSAGE_PREFIX + c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Luu chunks vao ChromaDB (persistent, cosine). Tao lai collection moi lan chay
    de dam bao idempotent (khong nhan doi du lieu).
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collection de re-index sach
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, embeddings, metadatas = [], [], [], []
    for c in chunks:
        src = c["metadata"]["source"]
        idx = c["metadata"]["chunk_index"]
        ids.append(f"{src}__{idx}")
        documents.append(c["content"])
        embeddings.append(c["embedding"])
        metadatas.append(c["metadata"])

    # Chroma gioi han batch ~5461 object/lan - corpus nho nen add 1 lan la du,
    # van chia batch cho an toan.
    BATCH = 2000
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            documents=documents[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
    return collection


def run_pipeline():
    """Chay toan bo pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} -> {CHROMA_DIR}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks (dim={len(chunks[0]['embedding']) if chunks else 0})")

    index_to_vectorstore(chunks)
    print(f"[OK] Indexed {len(chunks)} chunks to collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    run_pipeline()
