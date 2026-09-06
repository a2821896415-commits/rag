# -*- coding: utf-8 -*-
"""
理解能力评测模块。

核心思路：直接复用线上真实的主体确认链路（item_name_confirm_service），而非另写一套意图逻辑，
从而保证测的是"系统真实的意图识别 / 实体抽取 / 澄清与拒答决策"能力。

链路（与线上同源，去掉历史落库副作用）：
  rewrite_query_and_extract_item_names  -> 一次 LLM 调用产出 rewritten_query / item_names / is_chitchat
  search_item_name_candidates          -> 到 item_name 向量库找相近主体
  select_item_names                    -> 阈值判定（>=0.55 确认 / 0.40~0.55 候选待澄清 / 否则拒答）
  apply_item_name_result               -> 写回 state（确认则继续；候选则澄清反问；无则拒答；闲聊则兜底）

断言维度：
  intent_accuracy  : 预测 is_chitchat 是否与 expected_intent.is_chitchat 一致
  entity_accuracy  : 确认主体名集合与 expected_item_names 的 Jaccard 相似度（部分分，避免同义/多书误判）
  behavior_accuracy: 推断出的系统行为（answer/clarify/refuse/chitchat）是否命中 expected_behavior
  comprehension_score: 三者均值（可加权）

行为推断规则：
  - is_chitchat 且无抽取到主体           -> chitchat
  - 确认到主体                          -> answer
  - apply 后 answer 含"您是想问以下哪本书" -> clarify
  - apply 后 answer 含"没有找到相关书籍"   -> refuse
  - 其余                                -> 按 is_chitchat 退化
"""
from typing import Any

from app.rag.query import item_name_confirm_service as confirm
from evaluation.metrics import mean

# 与 item_name_confirm_service 保持一致的行为判定关键词
_CLARIFY_HINT = "您是想问以下哪本书"
_REFUSE_HINT = "没有找到相关书籍"

# 主体确认链路只能产出 {answer, clarify, refuse, chitchat} 四种行为。
# 在"主体确认"层，refuse_book_content（作者/实体在库但该书内容不在库）与 refuse 不可区分
# ——内容是否缺失由检索/生成层判定，故在理解能力评测中归一为 refuse。
_BEHAVIOR_CANON = {
    "answer": "answer",
    "clarify": "clarify",
    "refuse": "refuse",
    "refuse_book_content": "refuse",
    "chitchat": "chitchat",
}
# 拒绝类行为（库外 / 库内无内容）：被拒书籍本就不在主体库，无法被确认，
# 故实体准确率不参与理解综合分（置 None，报告聚合时跳过）。
_REFUSAL_BEHAVIORS = {"refuse", "refuse_book_content"}


def _run_confirm_chain(query: str, history: list[dict] | None = None) -> dict:
    """在内存 state 上跑真实确认链路（不落库历史），返回决策画像。"""
    history = history or []
    llm_result = confirm.rewrite_query_and_extract_item_names(history, query)
    item_names = llm_result.get("item_names", []) or []
    rewritten = llm_result.get("rewritten_query") or query
    is_chitchat = bool(llm_result.get("is_chitchat"))

    # 闲聊且未抽到主体：与线上 confirm_item_name 一致直接收口，不进检索。
    if is_chitchat and not item_names:
        return {
            "is_chitchat": True,
            "confirmed": [],
            "answer": confirm.CHITCHAT_FALLBACK,
            "rewritten_query": query,
            "behavior": "chitchat",
        }

    state: dict[str, Any] = {"original_query": query, "session_id": "__eval__", "history": history}
    final: dict[str, Any] = {}
    if item_names:
        final = confirm.select_item_names(confirm.search_item_name_candidates(item_names))
    confirm.apply_item_name_result(state, final, rewritten)

    answer = state.get("answer", "")
    confirmed = state.get("item_names", []) or []
    if confirmed:
        behavior = "answer"
    elif _CLARIFY_HINT in answer:
        behavior = "clarify"
    elif _REFUSE_HINT in answer:
        behavior = "refuse"
    else:
        behavior = "chitchat" if is_chitchat else "answer"
    return {
        "is_chitchat": is_chitchat,
        "confirmed": confirmed,
        "answer": answer,
        "rewritten_query": rewritten,
        "behavior": behavior,
    }


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 4)


def evaluate_case_comprehension(case: dict, session_id: str = "__eval__") -> dict:
    """评测单条 case 的理解能力。

    返回 intent_accuracy / entity_accuracy / behavior_accuracy / comprehension_score / 决策画像。
    """
    query = case.get("query", "")
    history = case.get("history") or None
    exp_intent = case.get("expected_intent", {}) or {}
    exp_chitchat = bool(exp_intent.get("is_chitchat", False))
    exp_items = case.get("expected_item_names", []) or []
    exp_behavior = case.get("expected_behavior", "answer")
    # 归一：refuse_book_content 在主体确认层等价于 refuse
    exp_behavior_canon = _BEHAVIOR_CANON.get(exp_behavior, exp_behavior)
    is_refusal = exp_behavior in _REFUSAL_BEHAVIORS

    try:
        dec = _run_confirm_chain(query, history)
    except Exception as exc:  # noqa: BLE001
        return {
            "case_id": case.get("case_id"),
            "intent_accuracy": None,
            "entity_accuracy": None,
            "behavior_accuracy": None,
            "comprehension_score": None,
            "error": f"confirm chain failed: {exc}",
        }

    intent_acc = 1.0 if (dec["is_chitchat"] == exp_chitchat) else 0.0
    entity_acc = _jaccard(dec["confirmed"], exp_items)
    behavior_acc = 1.0 if dec["behavior"] == exp_behavior_canon else 0.0
    # 拒绝类：被拒书籍本就不在主体库，实体无法被确认，故实体准确率不计入综合分
    if is_refusal:
        entity_acc = None

    parts = [intent_acc, behavior_acc]
    if entity_acc is not None:
        parts.append(entity_acc)
    score = round(mean(parts), 4)

    return {
        "case_id": case.get("case_id"),
        "is_chitchat_pred": dec["is_chitchat"],
        "is_chitchat_exp": exp_chitchat,
        "confirmed_pred": dec["confirmed"],
        "expected_item_names": exp_items,
        "behavior_pred": dec["behavior"],
        "expected_behavior": exp_behavior,
        "answer_snippet": (dec["answer"] or "")[:80],
        "intent_accuracy": intent_acc,
        "entity_accuracy": entity_acc,
        "behavior_accuracy": behavior_acc,
        "comprehension_score": score,
    }
