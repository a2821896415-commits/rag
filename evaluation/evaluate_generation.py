"""
生成评测模块（答案忠实度 / 相关性 / 要点覆盖）。

做法：
- 复用线上 `build_answer_prompt` 组装 Prompt，用同一套 LLM 直接产出答案，
  保证评测的是"系统真实生成逻辑"而非另写一套。
- 再由 LLM 裁判（json_mode）对 (query, 检索上下文, answer) 打分：
    faithfulness  : 答案是否被检索上下文支持、有无编造（0-1）
    relevance    : 答案是否正面回应了问题（0-1）
    fact_coverage: case 标注的 hard_facts 被答案覆盖/支持的比例（0-1）
"""
import json
from typing import Any

from app.infra.llm import llm_provider
from app.rag.query.answer_output_service import build_answer_prompt, NO_RELEVANT_ANSWER

# 参与生成与评测时使用的角色视角（默认听众）
DEFAULT_ROLE = "听众"
# 喂给裁判的上下文最多取前 N 个切块，控制 token 成本
JUDGE_CONTEXT_K = 5


def _to_reranked_docs(retrieved: list[dict]) -> list[dict]:
    """把 search_chunks 的结果映射为 produce_answer 期望的 reranked_docs 结构（content→text）。"""
    docs = []
    for c in retrieved:
        docs.append({
            "chunk_id": c.get("chunk_id"),
            "type": c.get("type", "milvus"),
            "title": c.get("title"),
            "text": c.get("content", ""),
            "score": c.get("score", 0.0),
            "book_name": c.get("book_name", ""),
            "author": c.get("author", ""),
            "entry_name": c.get("entry_name", ""),
            "content_type": c.get("content_type", ""),
            "file_title": c.get("file_title", ""),
            "url": c.get("url"),
        })
    return docs


def generate_answer(
    query: str,
    item_names: list[str],
    retrieved: list[dict],
    role: str = DEFAULT_ROLE,
) -> str:
    """用系统真实 Prompt 模板产出答案。检索为空时返回系统兜底语。"""
    reranked = _to_reranked_docs(retrieved)
    if not reranked:
        return NO_RELEVANT_ANSWER
    prompt = build_answer_prompt(reranked, query, item_names, [], role=role)
    client = llm_provider.chat()
    resp = client.invoke(prompt)
    return resp.content or ""


_JUDGE_PROMPT = """你是一名严格的听书知识库问答质量评审。请根据【检索上下文】评判【回答】的质量，只输出 JSON，不要输出其它内容。

【用户问题】
{query}

【检索上下文】
{context}

【标准要点（hard_facts，回答应尽可能覆盖/支持）】
{facts}

请评估：
1. faithfulness（0-1）：回答是否完全由检索上下文支持，有无编造、歪曲或无依据推断。完全支持=1，明显编造=0。
2. relevance（0-1）：回答是否正面、完整地回应了用户问题。完全回应=1，答非所问=0。
3. fact_coverage（0-1）：标准要点中有多少比例被回答覆盖或支持。全部覆盖=1，几乎未覆盖=0。

输出格式（严格 JSON）：
{{"faithfulness": <float>, "relevance": <float>, "fact_coverage": <float>, "reason": "<简要理由>"}}"""


def _parse_judge(raw: str) -> dict:
    """从 LLM 输出中稳妥地解析 JSON 评测结果。"""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        # 退化：尝试提取第一个 {...}
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
    return {"faithfulness": None, "relevance": None, "fact_coverage": None, "reason": "解析失败"}


def judge(
    query: str,
    answer: str,
    context_text: str,
    hard_facts: list[str] | None = None,
) -> dict:
    """调用 LLM 裁判，返回 faithfulness / relevance / fact_coverage / reason。"""
    facts = "\n".join(f"- {f}" for f in (hard_facts or [])) or "（无）"
    prompt = _JUDGE_PROMPT.format(query=query, context=context_text, facts=facts)
    client = llm_provider.chat(json_mode=True)
    resp = client.invoke(prompt)
    return _parse_judge(resp.content)


def evaluate_case_generation(
    case: dict,
    retrieved: list[dict],
    role: str = DEFAULT_ROLE,
) -> dict[str, Any]:
    """评测单条 case 的生成质量。

    仅在 run_eval 判定"应正常作答"的 case 上调用（refuse/clarify/negative 另行处理）。
    """
    query = case.get("query", "")
    item_names = (
        case.get("expected_item_names")
        or case.get("expect_source_books")
        or sorted({g.get("item_name") for g in case.get("golden_chunks", []) if g.get("item_name")})
    )
    answer = generate_answer(query, item_names, retrieved, role=role)

    top_context = retrieved[:JUDGE_CONTEXT_K]
    context_text = "\n\n".join(
        f"[{i + 1}] {c.get('content', '')}" for i, c in enumerate(top_context)
    )
    hard_facts = case.get("hard_facts") or []

    j = judge(query, answer, context_text, hard_facts)
    return {
        "case_id": case.get("case_id"),
        "answer": answer,
        "context_len": len(context_text),
        "faithfulness": j.get("faithfulness"),
        "relevance": j.get("relevance"),
        "fact_coverage": j.get("fact_coverage"),
        "judge_reason": j.get("reason"),
        "hard_facts_count": len(hard_facts),
    }
