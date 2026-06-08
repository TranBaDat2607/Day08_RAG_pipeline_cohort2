"""Cấu hình chung cho pipeline ingestion (Nguyễn Thành Đạt — 2A202600771)."""

from pathlib import Path

GROUP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = GROUP_ROOT / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
FAISS_DIR = DATA_DIR / "faiss"

FAISS_INDEX_FILE = FAISS_DIR / "index.faiss"
FAISS_META_FILE = FAISS_DIR / "metadata.pkl"
COLLECTION_NAME = "DrugLawDocs"

# RecursiveCharacterTextSplitter: ưu tiên cắt theo Chương/Điều trong văn bản luật VN
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
CHUNKING_METHOD = "recursive"

# E5-small: nhẹ, đa ngôn ngữ, khớp stack nhóm (prefix query:/passage: bắt buộc)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
EMBED_PASSAGE_PREFIX = "passage: "
EMBED_QUERY_PREFIX = "query: "

VECTOR_STORE = "faiss"
