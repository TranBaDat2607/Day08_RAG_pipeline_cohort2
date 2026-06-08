"""
RAG Evaluation Pipeline — DeepEval.

Đánh giá chất lượng pipeline RAG của nhóm (Task 9 retrieve + Task 10 generation)
trên golden dataset, sử dụng 4 metric của DeepEval và so sánh A/B giữa 2 cấu hình
retrieval.

Quy trình:
    1. Load golden_dataset.json (>= 15 cặp Q&A).
    2. Với MỖI config (A: hybrid + rerank, B: dense-only), chạy pipeline trên từng
       câu hỏi -> thu (answer, retrieval_context).
    3. Chấm 4 metric bằng LLM-as-judge (gpt-4o-mini):
         - Faithfulness        : answer có bám đúng context không?
         - Answer Relevancy     : answer có trả lời đúng câu hỏi không?
         - Contextual Recall    : context lấy về có đủ evidence cho expected_answer không?
         - Contextual Precision : trong context lấy về, bao nhiêu % thực sự hữu ích?
    4. So sánh A/B + phân tích worst performers + đề xuất cải tiến -> results.md.

Chạy:
    python -m group_project.evaluation.eval_pipeline
    python -m group_project.evaluation.eval_pipeline --limit 5   # chạy nhanh 5 câu

Yêu cầu: OPENAI_API_KEY trong group_project/.env (judge model + generation model).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --- Tắt telemetry của DeepEval cho gọn output -------------------------------
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

# --- Nạp .env (ưu tiên group_project/.env nơi nhóm để key) -------------------
from dotenv import load_dotenv  # noqa: E402

_EVAL_DIR = Path(__file__).resolve().parent
_GROUP_DIR = _EVAL_DIR.parent
_REPO_ROOT = _GROUP_DIR.parent
load_dotenv()
load_dotenv(_REPO_ROOT / ".env", override=True)
load_dotenv(_GROUP_DIR / ".env", override=True)

# --- Pipeline của nhóm -------------------------------------------------------
from group_project.core.task9_retrieval_pipeline import retrieve  # noqa: E402
from group_project.core.task5_semantic_search import semantic_search  # noqa: E402
from group_project.core.task10_generation import (  # noqa: E402
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    format_context,
    reorder_for_llm,
)

GOLDEN_DATASET_PATH = _EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = _EVAL_DIR / "results.md"
RESULTS_JSON_PATH = _EVAL_DIR / "results.json"

# Model dùng cho generation và cho judge. gpt-4o-mini: rẻ, đủ tốt cho eval.
GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
# Chặn completion tokens của judge: verdict JSON của DeepEval hiếm khi cần >2k token.
# Cap ở 4000 vừa thoải mái cho các metric dài (recall/precision liệt kê từng câu) vừa
# ngăn call chạy lố tới trần 16.384 token rồi retry.
JUDGE_MAX_TOKENS = 4000
TOP_K = 5
THRESHOLD = 0.7  # ngưỡng pass cho mỗi metric (theo gợi ý đề bài)

# Hai cấu hình retrieval để so sánh A/B.
CONFIGS = {
    "A_hybrid_rerank": {
        "label": "Hybrid (semantic ∥ lexical) + RRF + MMR rerank",
        "mode": "hybrid",
    },
    "B_dense_only": {
        "label": "Dense-only (semantic search, không lexical, không rerank)",
        "mode": "dense",
    },
}

METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# =============================================================================
# Load golden dataset
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Generation — tái dùng prompt/format của Task 10, nhưng cho phép truyền sẵn chunks
# =============================================================================

def _generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Sinh câu trả lời có citation từ chunks cho sẵn.

    Tách riêng (thay vì gọi generate_with_citation) để A/B test CHỈ thay đổi
    bước retrieval, còn bước generation (prompt, model, params) giữ y hệt nhau.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return response.choices[0].message.content or ""


def run_config(query: str, mode: str) -> tuple[str, list[dict]]:
    """
    Chạy 1 lượt pipeline theo cấu hình retrieval `mode`.

    - "hybrid": Task 9 đầy đủ (semantic ∥ lexical -> RRF -> MMR rerank -> fallback).
    - "dense" : chỉ semantic_search (Task 5), không lexical/rerank/fallback.

    Returns: (answer, chunks) với chunks = retrieval_context đã dùng.
    """
    if mode == "hybrid":
        chunks = retrieve(query, top_k=TOP_K, use_reranking=True)
    elif mode == "dense":
        chunks = semantic_search(query, top_k=TOP_K) or []
        for c in chunks:
            c.setdefault("source", "dense")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    answer = _generate_answer(query, chunks)
    return answer, chunks


# =============================================================================
# DeepEval — chấm 4 metric cho 1 test case
# =============================================================================

def _build_judge():
    """
    Judge LLM cho DeepEval, có CHẶN max_tokens (custom, không dùng GPTModel mặc định).

    Vì sao tự viết: `GPTModel` của DeepEval (bản 2.9.x) gọi
    `client.beta.chat.completions.parse(...)` mà KHÔNG truyền `max_tokens`. Với
    gpt-4o-mini, có verdict JSON sinh chạy tới trần 16.384 token rồi bị cắt giữa
    chừng -> "Could not parse response content as the length limit was reached" ->
    retry chậm/treo. Lớp dưới đây truyền thẳng `max_tokens` và `temperature=0` vào
    lời gọi OpenAI nên verdict luôn kết thúc gọn và ổn định (deterministic).
    """
    from deepeval.models.base_model import DeepEvalBaseLLM

    class CappedGPTJudge(DeepEvalBaseLLM):
        def __init__(self, model: str = JUDGE_MODEL, max_tokens: int = JUDGE_MAX_TOKENS):
            self.model_name = model
            self.max_tokens = max_tokens
            super().__init__(model)

        def load_model(self):
            from openai import OpenAI

            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def get_model_name(self) -> str:
            return self.model_name

        def generate(self, prompt: str, schema=None):
            client = self.load_model()
            if schema is not None:
                comp = client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema,
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                # Custom DeepEvalBaseLLM: metric dùng thẳng giá trị trả về (không
                # unpack (obj, cost) như GPTModel), nên trả luôn object đã parse.
                return comp.choices[0].message.parsed
            comp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=self.max_tokens,
            )
            return comp.choices[0].message.content

        async def a_generate(self, prompt: str, schema=None):
            # Chạy đồng bộ (async_mode=False) nên hàm này hiếm khi được gọi.
            return self.generate(prompt, schema)

    return CappedGPTJudge()


def _build_metrics():
    """Khởi tạo 4 metric của DeepEval (judge = gpt-4o-mini cap token, chạy đồng bộ)."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
    )

    judge = _build_judge()
    common = dict(model=judge, threshold=THRESHOLD, async_mode=False)
    return {
        "faithfulness": FaithfulnessMetric(**common),
        "answer_relevancy": AnswerRelevancyMetric(**common),
        "context_recall": ContextualRecallMetric(**common),
        "context_precision": ContextualPrecisionMetric(**common),
    }


def score_test_case(metrics: dict, item: dict, answer: str, contexts: list[str]) -> dict:
    """
    Chấm 1 test case với cả 4 metric.

    Returns dict {metric_key: {"score": float|None, "reason": str}}.
    Lỗi ở 1 metric (vd judge timeout) không làm hỏng cả run -> score=None.
    """
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=answer,
        expected_output=item["expected_answer"],
        retrieval_context=contexts if contexts else [""],
    )

    out = {}
    for key, metric in metrics.items():
        try:
            metric.measure(test_case)
            out[key] = {"score": metric.score, "reason": (metric.reason or "")[:500]}
        except Exception as e:  # noqa: BLE001
            out[key] = {"score": None, "reason": f"[ERROR] {type(e).__name__}: {e}"}
    return out


# =============================================================================
# Chạy evaluation cho 1 config trên toàn dataset
# =============================================================================

def evaluate_config(config_key: str, golden_dataset: list[dict]) -> dict:
    """Chạy pipeline + chấm điểm cho 1 config trên toàn bộ golden dataset."""
    cfg = CONFIGS[config_key]
    print(f"\n{'='*72}\n[CONFIG {config_key}] {cfg['label']}\n{'='*72}")

    metrics = _build_metrics()
    per_case = []

    for i, item in enumerate(golden_dataset, 1):
        q = item["question"]
        print(f"  ({i}/{len(golden_dataset)}) {q[:60]}...")
        answer, chunks = run_config(q, cfg["mode"])
        contexts = [c.get("content", "") for c in chunks]
        scores = score_test_case(metrics, item, answer, contexts)

        per_case.append(
            {
                "question": q,
                "expected_answer": item["expected_answer"],
                "expected_context": item.get("expected_context", ""),
                "answer": answer,
                "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
                "num_chunks": len(chunks),
                "scores": scores,
            }
        )
        line = " | ".join(
            f"{k.split('_')[0][:4]}={(scores[k]['score'] if scores[k]['score'] is not None else float('nan')):.2f}"
            for k in METRIC_KEYS
        )
        print(f"        -> {line}")

    return {"config_key": config_key, "label": cfg["label"], "cases": per_case}


def _aggregate(cases: list[dict]) -> dict:
    """Trung bình mỗi metric (bỏ qua None) + pass-rate theo threshold."""
    agg = {}
    for key in METRIC_KEYS:
        vals = [c["scores"][key]["score"] for c in cases if c["scores"][key]["score"] is not None]
        avg = sum(vals) / len(vals) if vals else 0.0
        passed = sum(1 for v in vals if v >= THRESHOLD)
        agg[key] = {"avg": avg, "pass_rate": passed / len(vals) if vals else 0.0, "n": len(vals)}
    overall = [agg[k]["avg"] for k in METRIC_KEYS]
    agg["overall_avg"] = sum(overall) / len(overall) if overall else 0.0
    return agg


# =============================================================================
# Export report -> results.md
# =============================================================================

def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else str(x)


def export_results(results: dict):
    """Ghi bảng điểm + A/B + worst performers + recommendations ra results.md."""
    configs = results["configs"]
    keys = list(configs.keys())
    metric_titles = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    md = []
    md.append("# RAG Evaluation Results — DeepEval\n")
    md.append(f"- **Framework**: DeepEval (LLM-as-judge = `{JUDGE_MODEL}`)")
    md.append(f"- **Generation model**: `{GEN_MODEL}`, temperature={TEMPERATURE}, top_p={TOP_P}")
    md.append(f"- **Golden dataset**: {results['n_cases']} cặp Q&A (legal + news corpus)")
    md.append(f"- **top_k**: {TOP_K} · **pass threshold**: {THRESHOLD}")
    md.append("")
    md.append(
        "4 metric: **Faithfulness** (answer bám context?), **Answer Relevancy** "
        "(answer đúng câu hỏi?), **Context Recall** (retriever lấy đủ evidence?), "
        "**Context Precision** (% context hữu ích, đúng thứ hạng?).\n"
    )

    # --- Bảng A/B tổng hợp ---------------------------------------------------
    md.append("## 1. Bảng điểm tổng hợp & So sánh A/B\n")
    header = "| Metric | " + " | ".join(configs[k]["label_short"] for k in keys) + " | Δ (A−B) |"
    sep = "|" + "---|" * (len(keys) + 2)
    md.append(header)
    md.append(sep)
    for mk in METRIC_KEYS:
        row = [metric_titles[mk]]
        vals = []
        for k in keys:
            a = configs[k]["agg"][mk]
            row.append(f"{_fmt(a['avg'])} ({a['pass_rate']*100:.0f}% pass)")
            vals.append(a["avg"])
        delta = vals[0] - vals[1] if len(vals) >= 2 else 0.0
        row.append(f"{delta:+.3f}")
        md.append("| " + " | ".join(row) + " |")
    # overall
    ov = [configs[k]["agg"]["overall_avg"] for k in keys]
    od = (ov[0] - ov[1]) if len(ov) >= 2 else 0.0
    md.append(
        "| **Overall avg** | "
        + " | ".join(f"**{_fmt(v)}**" for v in ov)
        + f" | **{od:+.3f}** |"
    )
    md.append("")
    md.append("> Config A = " + configs[keys[0]]["label"])
    md.append(">")
    md.append("> Config B = " + configs[keys[1]]["label"])
    md.append("")

    winner = keys[0] if ov[0] >= ov[1] else keys[1]
    md.append(
        f"**Kết luận A/B:** Config **{winner}** "
        f"({configs[winner]['label_short']}) đạt overall cao hơn "
        f"({_fmt(max(ov))} vs {_fmt(min(ov))}).\n"
    )

    # --- Worst performers cho config thắng -----------------------------------
    md.append("## 2. Phân tích Worst Performers\n")
    for k in keys:
        cases = configs[k]["cases"]
        scored = []
        for c in cases:
            vs = [c["scores"][m]["score"] for m in METRIC_KEYS if c["scores"][m]["score"] is not None]
            scored.append((sum(vs) / len(vs) if vs else 0.0, c))
        scored.sort(key=lambda t: t[0])
        md.append(f"### Config {k} — {configs[k]['label_short']}\n")
        md.append("| Câu hỏi | Avg | Faith | Rel | Recall | Prec | Vấn đề chính |")
        md.append("|---|---|---|---|---|---|---|")
        for avg, c in scored[:3]:
            s = c["scores"]
            def g(m):
                v = s[m]["score"]
                return f"{v:.2f}" if v is not None else "n/a"
            # tìm metric tệ nhất để chú thích
            worst_m = min(
                (m for m in METRIC_KEYS if s[m]["score"] is not None),
                key=lambda m: s[m]["score"],
                default=None,
            )
            note = (s[worst_m]["reason"][:140] + "…") if worst_m else "—"
            note = note.replace("\n", " ").replace("|", "/")
            md.append(
                f"| {c['question'][:55]}… | {avg:.2f} | {g('faithfulness')} | "
                f"{g('answer_relevancy')} | {g('context_recall')} | {g('context_precision')} | {note} |"
            )
        md.append("")

    # --- Recommendations -----------------------------------------------------
    md.append("## 3. Đề xuất cải tiến\n")
    md.append(_recommendations(configs, keys))
    md.append("")

    # --- Chi tiết từng case (config thắng) -----------------------------------
    md.append(f"## 4. Chi tiết từng câu hỏi (Config {winner})\n")
    for i, c in enumerate(configs[winner]["cases"], 1):
        s = c["scores"]
        md.append(f"**Q{i}. {c['question']}**\n")
        md.append(f"- *Retrieval source*: `{c['retrieval_source']}` · *chunks*: {c['num_chunks']}")
        md.append(
            "- *Scores*: "
            + ", ".join(
                f"{metric_titles[m]}={_fmt(s[m]['score']) if s[m]['score'] is not None else 'n/a'}"
                for m in METRIC_KEYS
            )
        )
        ans = c["answer"].replace("\n", " ")
        md.append(f"- *Answer*: {ans[:300]}{'…' if len(ans) > 300 else ''}")
        md.append("")

    RESULTS_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\n✓ Report -> {RESULTS_PATH}")


def _recommendations(configs: dict, keys: list[str]) -> str:
    """Sinh đề xuất cải tiến dựa trên metric yếu nhất (data-driven)."""
    # lấy config tốt nhất
    winner = max(keys, key=lambda k: configs[k]["agg"]["overall_avg"])
    agg = configs[winner]["agg"]
    recs = []

    # Xếp hạng metric từ yếu -> mạnh
    ranked = sorted(METRIC_KEYS, key=lambda m: agg[m]["avg"])
    tips = {
        "faithfulness": (
            "**Faithfulness thấp** → answer có chi tiết không nằm trong context (hallucination). "
            "Siết SYSTEM_PROMPT ('chỉ dùng thông tin trong context, không suy diễn'), hạ temperature, "
            "và thêm bước hậu kiểm: yêu cầu LLM trích nguyên văn câu hỗ trợ cho mỗi citation."
        ),
        "answer_relevancy": (
            "**Answer Relevancy thấp** → câu trả lời lan man / lạc đề. "
            "Yêu cầu trả lời trực tiếp, ngắn gọn trước khi diễn giải; cắt các đoạn không liên quan câu hỏi."
        ),
        "context_recall": (
            "**Context Recall thấp** → retriever bỏ sót evidence. Tăng top_k / fetch_k, "
            "cải thiện chunking (tách theo Điều/Khoản để 1 chunk = 1 đơn vị pháp lý trọn vẹn), "
            "thử embedding mạnh hơn (bge-m3 / e5-large) và query expansion cho câu hỏi tiếng Việt."
        ),
        "context_precision": (
            "**Context Precision thấp** → chunk hữu ích không được xếp lên đầu / lẫn nhiễu. "
            "Thêm cross-encoder reranker (Jina/Qwen) thay vì chỉ MMR, giảm top_k, "
            "và lọc chunk rác (mục lục, footer báo, danh sách link) ở bước chuẩn hoá."
        ),
    }
    recs.append("Xếp theo mức độ ưu tiên (metric yếu nhất trước):\n")
    for i, m in enumerate(ranked, 1):
        recs.append(f"{i}. {tips[m]} *(điểm hiện tại: {agg[m]['avg']:.3f})*")

    # So sánh A/B -> đề xuất chọn config
    ov = {k: configs[k]["agg"]["overall_avg"] for k in keys}
    recs.append("")
    recs.append(
        f"**Lựa chọn cấu hình:** dùng **{winner}** ({configs[winner]['label_short']}) "
        f"làm mặc định production vì overall cao nhất ({ov[winner]:.3f}). "
        "Các cấu hình còn lại có thể dùng làm fallback hoặc cho câu hỏi đặc thù."
    )
    recs.append("")
    recs.append(
        "**Mở rộng dataset:** golden set hiện thiên về câu hỏi factual đơn lẻ; nên bổ sung "
        "câu hỏi multi-hop (so sánh nhiều Điều luật), câu hỏi không có trong corpus (kiểm tra "
        "khả năng từ chối 'không xác minh được'), và câu hỏi follow-up để đánh giá conversation memory."
    )
    return "\n".join(recs)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation với DeepEval")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số câu hỏi (debug)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("✗ Thiếu OPENAI_API_KEY trong .env — không thể chạy generation + judge.")
        sys.exit(1)

    golden_dataset = load_golden_dataset()
    if args.limit:
        golden_dataset = golden_dataset[: args.limit]
    print(f"Loaded {len(golden_dataset)} test cases từ golden_dataset.json")

    configs_out = {}
    for ck in CONFIGS:
        res = evaluate_config(ck, golden_dataset)
        agg = _aggregate(res["cases"])
        configs_out[ck] = {
            "label": CONFIGS[ck]["label"],
            "label_short": ck.split("_", 1)[1].replace("_", " "),
            "agg": agg,
            "cases": res["cases"],
        }
        print(f"  [agg {ck}] overall={agg['overall_avg']:.3f}")

    results = {"n_cases": len(golden_dataset), "configs": configs_out}

    # Lưu JSON thô để tái lập / phân tích thêm.
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✓ Raw results -> {RESULTS_JSON_PATH}")

    export_results(results)


if __name__ == "__main__":
    main()
