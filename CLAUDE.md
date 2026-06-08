# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A course assignment (VinAI cohort, Day 8) for building an end-to-end Vietnamese RAG pipeline over **Vietnamese drug laws + news articles about artists involved with drugs**. The `src/` modules ship as **stubs**: every function `raise NotImplementedError(...)` and carries the intended implementation as commented-out example code in its docstring/body. The task is to fill them in. `README.md` (Vietnamese) is the full assignment spec and grading rubric; `group_project/README.md` covers the team chatbot + evaluation deliverable.

When implementing a task, the commented example code in the stub is the recommended approach — but the **enforced contract is the test suite**, not the comments.

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env          # then fill in API keys

# Grading test suite (50 pts, one TestCase class per task)
pytest tests/ -v
pytest tests/test_individual.py::TestTask5 -v      # single task
pytest tests/test_individual.py::TestTask5::test_results_sorted_descending -v

# Run a single task module (each has a __main__ demo block).
# Run as a package from repo root — task9 uses relative imports (from .task5...):
python -m src.task4_chunking_indexing
python -m src.task9_retrieval_pipeline

# Group project (after individual tasks done)
python -m group_project.evaluation.eval_pipeline
streamlit run app.py          # app.py is not yet created — group deliverable
```

## Testing model — read before trusting a green run

Tests are `unittest` classes run under `pytest`. They are written to **`self.skipTest(...)` on `ImportError` / `NotImplementedError`** rather than fail. Consequences:

- A fully-passing run with everything **skipped** looks identical at a glance to a passing run with everything **implemented**. Always check skip counts / `-v` output to know which tasks are actually done.
- Several tests also skip when data is absent (no documents indexed) or an API key is missing. "Passing" ≠ "implemented and exercised."
- Tests `import` from the `src.taskN_...` path, so implementations must keep their module names and live in `src/`.

## Architecture & data flow

The 10 tasks form one linear pipeline; data moves through `data/` stages and each `src/taskN_*.py` consumes the previous task's output:

```
data/landing/{legal,news}/   ← Task 1 (download PDF/DOCX), Task 2 (crawl articles)
        │ Task 3 (MarkItDown convert)
        ▼
data/standardized/{legal,news}/*.md
        │ Task 4: load_documents → chunk_documents → embed_chunks → index_to_vectorstore
        ▼
   vector store (Weaviate default)
        │
        ├─ Task 5  semantic_search(query, top_k)   → dense results
        ├─ Task 6  lexical_search(query, top_k)    → BM25 results
        │            │ Task 9 retrieve(): RRF-merge (rerank_rrf), then Task 7 rerank()
        │            ▼
        │       if best score < SCORE_THRESHOLD (0.3):
        └─ Task 8  pageindex_search()  ← vectorless fallback
                     ▼
        Task 10 generate_with_citation():
          reorder_for_llm (avoid "lost in the middle") → format_context → LLM → cited answer
```

### The dict-shape contract (enforced by tests — keep these exact keys)

Every retrieval function returns `list[dict]`. Required keys:

- Tasks 5/6/7: `{"content": str, "score": float, "metadata": dict}`, **sorted by `score` descending**, length ≤ `top_k`.
- Task 8 (`pageindex_search`): results carry `"source": "pageindex"`.
- Task 9 (`retrieve`): results carry `"source"` ∈ `{"hybrid", "pageindex"}` in addition to `content`/`score`. Must not crash when hybrid returns nothing (fallback path).
- Task 10: `generate_with_citation(query)` returns a `dict` with an `"answer"` str; `reorder_for_llm(chunks)` preserves chunk count and keeps the highest-scored chunk first; `format_context(chunks)` must embed `metadata["source"]` so citations resolve.

### Config lives as module-level constants (and is graded)

Task 4 exposes `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKING_METHOD`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `VECTOR_STORE`; Task 9 exposes `SCORE_THRESHOLD`, `DEFAULT_TOP_K`, `RERANK_METHOD`. `TestTask4.test_config_documented` asserts `0 < CHUNK_OVERLAP < CHUNK_SIZE`. The rubric expects code comments justifying each choice (which chunker, which embedding model, which store, and why).

### Reranking variants (Task 7)

`rerank()` dispatches on `method`: `cross_encoder` (Jina API / local Qwen), `mmr` (needs a `query_embedding`, called directly), `rrf` (`rerank_rrf(ranked_lists, ...)`, fuses multiple ranked lists and is what Task 9 uses to merge dense+sparse). Only `cross_encoder` routes through the unified `rerank()` entry point; `mmr`/`rrf` are called directly.

## Conventions

- **Language**: docstrings, comments, and print output are Vietnamese; the data corpus is Vietnamese — pick multilingual models (default stub uses `BAAI/bge-m3` for embeddings, Jina multilingual reranker).
- **Secrets**: all via `.env` (`OPENAI_API_KEY`, `JINA_API_KEY`, `PAGEINDEX_API_KEY`, `WEAVIATE_URL`/`WEAVIATE_API_KEY`). Load with `python-dotenv`.
- External services the pipeline can depend on: Weaviate (local Docker or Cloud), PageIndex (pageindex.ai account), and an LLM/reranker API. Tests are written to skip gracefully when these aren't configured.
