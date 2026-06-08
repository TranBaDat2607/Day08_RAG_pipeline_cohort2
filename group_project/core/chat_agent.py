"""
Chat Agent — Router hội thoại bằng OpenAI tool calling (function calling).

Mục tiêu: chatbot vừa nói chuyện tự nhiên, vừa biết KHI NÀO cần tra cứu pháp luật.

    - Câu xã giao / hỏi chung chung ("xin chào", "bạn là ai", "cảm ơn"...) → LLM trả
      lời trực tiếp bằng kiến thức của nó, KHÔNG chạy RAG. retrieval_source = "chat".
    - Câu hỏi về pháp luật ma tuý / quy định / nghệ sĩ liên quan ma tuý → LLM tự
      GỌI TOOL `search_drug_law`. Khi đó ta chạy pipeline retrieval của nhóm (Task 9
      hybrid + fallback), nạp các đoạn tìm được vào hội thoại, rồi để LLM viết câu
      trả lời CÓ CITATION dựa trên đúng evidence đó. retrieval_source = "hybrid"/"pageindex".

Vì sao dùng tool calling thay vì if/else từ khoá?
    - LLM hiểu ngữ cảnh & follow-up tốt hơn rule cứng (vd "thế còn vận chuyển thì sao?"
      vẫn được nhận diện là câu hỏi pháp luật nhờ lịch sử hội thoại).
    - Bản thân model quyết định query để tra cứu (có thể viết lại câu hỏi cho gọn) —
      đúng tinh thần agentic RAG.

Hàm public:
    chat(query, history=None, top_k=5) -> {answer, sources, retrieval_source}
    (cùng contract với generate_with_citation để backend/eval dùng thay thế được.)

Lưu ý: generate_with_citation (Task 10) vẫn giữ nguyên — là "đường RAG thuần" cho
evaluation. chat_agent chỉ thêm một lớp định tuyến ở trên.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Nạp .env: ưu tiên group_project/.env, rồi repo root (override biến shell cũ).
_GROUP_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _GROUP_DIR.parent
load_dotenv()
load_dotenv(_REPO_ROOT / ".env", override=True)
load_dotenv(_GROUP_DIR / ".env", override=True)

from .task9_retrieval_pipeline import retrieve
from .task10_generation import TEMPERATURE, TOP_K, TOP_P, format_context, reorder_for_llm


# =============================================================================
# CONFIG
# =============================================================================

MODEL = "gpt-4o-mini"

# System prompt: định nghĩa vai trò + LUẬT định tuyến tool. Nói rõ khi có context
# từ tool thì phải trích dẫn; câu xã giao thì trả lời trực tiếp, ngắn gọn.
SYSTEM_PROMPT = """Bạn là trợ lý ảo "LuậtMaTuý AI" — chuyên hỗ trợ về pháp luật phòng,
chống ma tuý tại Việt Nam (Luật PCMT 2021, Bộ luật Hình sự, nghị định/thông tư) và
tin tức liên quan đến nghệ sĩ dính líu ma tuý.

QUY TẮC ĐỊNH TUYẾN (rất quan trọng):
- Với MỌI câu hỏi liên quan đến pháp luật, quy định, hình phạt, thủ tục, hoặc tin tức
  về ma tuý/nghệ sĩ liên quan ma tuý → BẮT BUỘC gọi tool `search_drug_law` để tra cứu
  trong kho văn bản, KHÔNG tự bịa hay trả lời bằng trí nhớ.
- Với câu xã giao, chào hỏi, cảm ơn, hỏi về bản thân bạn, hoặc câu hỏi chung KHÔNG liên
  quan pháp luật ma tuý → trả lời trực tiếp, thân thiện, ngắn gọn bằng tiếng Việt, KHÔNG
  gọi tool. Nếu câu hỏi ngoài phạm vi (vd nấu ăn, thể thao) thì lịch sự nói rằng bạn
  chuyên về pháp luật ma tuý và mời người dùng hỏi về chủ đề đó.

KHI ĐÃ CÓ KẾT QUẢ TỪ TOOL (context):
- Chỉ dùng thông tin trong context được cung cấp.
- MỌI khẳng định/thông tin pháp lý PHẢI kèm citation trong ngoặc vuông, ví dụ
  [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [bo-luat-hinh-su-2015.md].
- Nếu context không đủ để trả lời, nói rõ "Tôi không thể xác minh thông tin này từ
  nguồn hiện có." thay vì đoán.
- Trình bày rõ ràng, có cấu trúc."""

# Định nghĩa tool cho function calling. Model sẽ tự điền 'query' (có thể viết lại
# câu hỏi cho gọn/đủ ngữ cảnh) khi quyết định cần tra cứu.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_drug_law",
            "description": (
                "Tra cứu kho văn bản pháp luật phòng chống ma tuý của Việt Nam và tin "
                "tức liên quan. Gọi hàm này cho mọi câu hỏi về luật, quy định, hình phạt, "
                "thủ tục, hoặc tin tức về ma tuý/nghệ sĩ liên quan ma tuý."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Câu truy vấn tra cứu bằng tiếng Việt, viết đầy đủ ngữ cảnh "
                            "(tự bổ sung từ lịch sử hội thoại nếu câu hỏi là follow-up)."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_messages(query: str, history: list[dict] | None) -> list[dict]:
    """Ghép system + lịch sử hội thoại + câu hỏi hiện tại thành messages."""
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        # UI dùng 'ai' cho assistant; OpenAI cần 'assistant'.
        if role in ("ai", "assistant"):
            messages.append({"role": "assistant", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
    messages.append({"role": "user", "content": query})
    return messages


def chat(query: str, history: list[dict] | None = None, top_k: int = TOP_K) -> dict:
    """
    Hội thoại có định tuyến RAG bằng tool calling.

    Args:
        query: Câu hỏi/câu nói của người dùng.
        history: Lịch sử hội thoại [{"role": "user"|"ai"|"assistant", "content": str}, ...].
        top_k: Số chunk tra cứu khi cần RAG.

    Returns:
        {
            'answer': str,
            'sources': list[dict],          # [] nếu trả lời chat thường
            'retrieval_source': str         # 'hybrid' | 'pageindex' | 'chat' | 'none'
        }
    """
    query = (query or "").strip()
    if not query:
        return {"answer": "Vui lòng nhập câu hỏi.", "sources": [], "retrieval_source": "none"}

    client = _get_client()
    messages = _build_messages(query, history)

    # --- Lượt 1: để model quyết định có gọi tool không -----------------------
    first = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    msg = first.choices[0].message

    # --- Không gọi tool -> trả lời hội thoại thường --------------------------
    if not msg.tool_calls:
        return {
            "answer": msg.content or "",
            "sources": [],
            "retrieval_source": "chat",
        }

    # --- Có gọi tool -> chạy RAG cho từng tool call --------------------------
    # Ghi lại message assistant chứa tool_calls (bắt buộc trước khi gửi tool result).
    messages.append({
        "role": "assistant",
        "content": msg.content or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ],
    })

    all_sources: list[dict] = []
    retrieval_source = "none"

    for tc in msg.tool_calls:
        # Lấy query do model sinh (fallback về câu hỏi gốc nếu parse lỗi).
        try:
            args = json.loads(tc.function.arguments or "{}")
            search_query = (args.get("query") or query).strip() or query
        except (json.JSONDecodeError, AttributeError):
            search_query = query

        if tc.function.name == "search_drug_law":
            chunks = retrieve(search_query, top_k=top_k)
        else:
            chunks = []

        if chunks:
            all_sources.extend(chunks)
            retrieval_source = chunks[0].get("source", "hybrid")
            # Reorder chống "lost in the middle" + format kèm nhãn nguồn để cite.
            tool_content = format_context(reorder_for_llm(chunks))
        else:
            tool_content = "Không tìm thấy tài liệu liên quan trong kho văn bản."

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_content,
        })

    # --- Lượt 2: model viết câu trả lời cuối có citation từ context ----------
    second = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = second.choices[0].message.content or ""

    if not all_sources:
        # Model gọi tool nhưng không có evidence -> để contract rõ ràng.
        retrieval_source = retrieval_source if retrieval_source != "none" else "none"

    return {
        "answer": answer,
        "sources": all_sources,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    demos = [
        "Xin chào, bạn là ai?",                                   # -> chat thường
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý?",      # -> RAG
        "Hôm nay trời đẹp nhỉ?",                                  # -> chat / từ chối lịch sự
    ]
    for q in demos:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        r = chat(q)
        print(f"[via {r['retrieval_source']} | {len(r['sources'])} sources]")
        print(r["answer"][:400])
