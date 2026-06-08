"""
group_project.core — Pipeline search → retrieval → generation của nhóm.

Merge từ bài cá nhân (Trần Bá Đạt — 2A202600778), nối lại với vector store FAISS
chung của nhóm (data/faiss/) do Nguyễn Thành Đạt dựng ở group_project/ingestion/.

Cùng embedding model intfloat/multilingual-e5-small (384d) + prefix query:/passage:
như bài cá nhân, nên chỉ đổi backend store ChromaDB -> FAISS và đường dẫn import.

Các module:
    task5_semantic_search   — dense retrieval trên FAISS
    task6_lexical_search    — BM25 (+ TF-IDF bonus) trên corpus của FAISS store
    task7_reranking         — MMR (chính) + RRF (gộp dense+sparse cho Task 9)
    task8_pageindex_vectorless — fallback vectorless (PageIndex Cloud)
    task9_retrieval_pipeline   — hybrid retrieve + fallback (entry point)
    task10_generation          — generate_with_citation (OpenAI gpt-4o-mini)
"""
