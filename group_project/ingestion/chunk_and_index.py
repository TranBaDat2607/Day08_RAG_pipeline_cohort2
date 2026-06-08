"""
Task 4 — Chunking & FAISS indexing (group project).

Pipeline: load_documents → chunk_documents → embed_chunks → index_to_faiss

Vector store: FAISS IndexFlatIP + L2-normalized embeddings (cosine similarity).
Metadata + nội dung chunk lưu tại data/faiss/metadata.pkl.
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKING_METHOD,
    COLLECTION_NAME,
    EMBED_PASSAGE_PREFIX,
    EMBED_QUERY_PREFIX,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_DIR,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    STANDARDIZED_DIR,
    VECTOR_STORE,
)

_store_cache: dict | None = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_documents() -> list[dict]:
    """Đọc markdown từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = normalize_text(md_file.read_text(encoding="utf-8"))
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\nChương ", "\nĐiều ", "\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for i, chunk_text in enumerate(splitter.split_text(doc["content"])):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i},
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return chunks

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


def index_to_faiss(chunks: list[dict]) -> None:
    """Lưu embeddings vào FAISS + metadata pickle."""
    import faiss

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    if not chunks:
        payload = {
            "chunks": [],
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "embed_query_prefix": EMBED_QUERY_PREFIX,
            "embed_passage_prefix": EMBED_PASSAGE_PREFIX,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "vector_store": VECTOR_STORE,
        }
        with open(FAISS_META_FILE, "wb") as f:
            pickle.dump(payload, f)
        return

    vectors = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # Không lưu embedding vào metadata để giảm dung lượng pickle
    stored_chunks = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    payload = {
        "chunks": stored_chunks,
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "embed_query_prefix": EMBED_QUERY_PREFIX,
        "embed_passage_prefix": EMBED_PASSAGE_PREFIX,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_store": VECTOR_STORE,
        "num_vectors": len(chunks),
    }
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(payload, f)

    global _store_cache
    _store_cache = {
        "index": index,
        "chunks": stored_chunks,
        **payload,
    }


def get_faiss_store() -> dict:
    """Load FAISS index + metadata (lazy cache)."""
    global _store_cache
    if _store_cache is not None:
        return _store_cache

    import faiss

    if FAISS_META_FILE.exists():
        with open(FAISS_META_FILE, "rb") as f:
            meta = pickle.load(f)
    else:
        meta = {
            "chunks": [],
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "embed_query_prefix": EMBED_QUERY_PREFIX,
        }

    index = None
    if FAISS_INDEX_FILE.exists():
        index = faiss.read_index(str(FAISS_INDEX_FILE))

    _store_cache = {"index": index, **meta}
    return _store_cache


def faiss_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Dense search trên FAISS (cosine qua inner product sau L2-normalize).

    Returns:
        List of {'content', 'score', 'metadata'} sorted descending.
    """
    store = get_faiss_store()
    index = store.get("index")
    chunks = store.get("chunks", [])
    if index is None or not chunks or index.ntotal == 0:
        return []

    import faiss
    from sentence_transformers import SentenceTransformer

    model_name = store.get("embedding_model", EMBEDDING_MODEL)
    query_prefix = store.get("embed_query_prefix", EMBED_QUERY_PREFIX)
    model = SentenceTransformer(model_name)

    query_vec = model.encode(
        query_prefix + query,
        normalize_embeddings=True,
    ).astype(np.float32)
    faiss.normalize_L2(query_vec.reshape(1, -1))

    scores, indices = index.search(query_vec.reshape(1, -1), min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[int(idx)]
        results.append({
            "content": chunk["content"],
            "score": float(score),
            "metadata": chunk.get("metadata", {}),
        })
    return results


def run_pipeline() -> None:
    print("=" * 60)
    print("Task 4: Chunking & FAISS Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} → {FAISS_DIR}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_faiss(chunks)
    print(f"✓ FAISS index: {FAISS_INDEX_FILE}")
    print(f"✓ Metadata  : {FAISS_META_FILE}")


if __name__ == "__main__":
    run_pipeline()
