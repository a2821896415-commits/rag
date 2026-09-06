"""
检索评测模块。

设计要点：
- 直接驱动真实检索入口 `search_chunks`（与线上同源的混合向量检索），
  仅用 case 标注的 `expected_item_names` 作为主体过滤，保持可控、可复现。
- golden → 检索结果匹配：
  * exact 模式：golden 与检索结果都带 `chunk_uid` 时，按 uid 精确判定命中。
  * soft  模式：否则用语义相似度（BGE-M3 稠密向量余弦）判断 evidence 是否命中，
    相似度 >= sim_threshold 视为相关。
- 输出各 k 下的 recall/precision/hit_rate/mrr/ndcg/f1，并额外给出
  golden 级覆盖率（soft 模式下比位置级 recall 更准确）。
"""
from typing import Any

import numpy as np

from app.infra.llm import llm_provider
from app.rag.query.search_embedding_service import search_chunks
from evaluation.metrics import compute_retrieval_metrics, mean

# soft 模式下判定"相关"的相似度阈值（BGE-M3 归一化余弦，[0,1]）
DEFAULT_SIM_THRESHOLD = 0.6
# 检索抓取的上限（用于后续按不同 k 计算指标）
DEFAULT_RETRIEVE_K = 10


def _embed(texts: list[str]) -> np.ndarray:
    """批量编码为稠密向量（已 L2 归一化，余弦=点积）。空输入返回空矩阵。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    out = llm_provider.embed_documents(texts)["dense"]
    return np.asarray(out, dtype=np.float32)


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回 shape=(len(a), len(b)) 的两两余弦相似度矩阵。"""
    if a.size == 0 or b.size == 0:
        return np.zeros((max(a.shape[0], 1), max(b.shape[0], 1)), dtype=np.float32)
    return a @ b.T


def retrieve(
    query: str,
    item_names: list[str],
    k: int = DEFAULT_RETRIEVE_K,
    use_rerank: bool = False,
) -> list[dict]:
    """调用真实检索入口，返回 top-k 切块（含 chunk_uid 字段）。

    Args:
        use_rerank: True 时额外套用线上重排 rerank_documents，
            让评测反映"真正喂给大模型"的排序；默认 False，测的是原始混合检索排序。
    """
    if not item_names:
        return []
    chunks = search_chunks(rewritten_query=query, item_names=item_names, limit=k)
    if not (use_rerank and chunks):
        return chunks

    # 重排是线上真实链路的一环，默认评测会跳过它，导致排序类指标（mrr/ndcg）偏悲观。
    from app.rag.query.rerank_service import rerank_documents

    by_id = {c.get("chunk_id"): c for c in chunks}
    reranked = rerank_documents(
        {"rrf_chunks": chunks, "web_search_docs": [], "rewritten_query": query}
    )
    # 重排产物把 content 改名成 text、且丢弃 chunk_uid，
    # 这里按 chunk_id 还原原始字段，保证 exact 精确匹配仍然可用。
    out: list[dict] = []
    for doc in reranked:
        src = by_id.get(doc.get("chunk_id"))
        if src is None:
            continue
        merged = dict(src)
        merged["score"] = doc.get("score", src.get("score", 0.0))
        out.append(merged)
    return out


def evaluate_case_retrieval(
    case: dict,
    k_list: list[int] | None = None,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    use_rerank: bool = False,
) -> dict[str, Any]:
    """评测单条 case 的检索质量。

    Returns:
        结果字典，含 mode / 各 k 指标 / golden 覆盖率等；
        若 case 无 golden_chunks 或无法判定相关性，mode 为 "no_golden_signal"。
    """
    k_list = k_list or [1, 3, 5, 10]
    retrieve_k = max(k_list + [retrieve_k])
    golden_chunks = case.get("golden_chunks", []) or []

    item_names = (
        case.get("expected_item_names")
        or case.get("expect_source_books")
        or sorted({g.get("item_name") for g in golden_chunks if g.get("item_name")})
    )

    result: dict[str, Any] = {
        "case_id": case.get("case_id"),
        "type": case.get("type"),
        "mode": "no_golden_signal",
        "rerank": use_rerank,
        "retrieved_count": 0,
        "golden_count": len(golden_chunks),
        "item_names": item_names,
        "metrics": {},
        "covered_golden_ratio": 0.0,
        "sim_threshold": sim_threshold,
        "error": None,
    }

    if not golden_chunks:
        result["mode"] = "no_golden"
        return result

    golden_uids = {g["chunk_uid"] for g in golden_chunks if g.get("chunk_uid")}
    golden_evidences = [g["evidence"] for g in golden_chunks if g.get("evidence")]

    try:
        retrieved = retrieve(
            case.get("query", ""), item_names, k=retrieve_k, use_rerank=use_rerank
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"retrieve failed: {exc}"
        return result

    result["retrieved_count"] = len(retrieved)
    result["retrieved"] = retrieved
    if not retrieved:
        result["mode"] = "empty_retrieval"
        result["metrics"] = {k: _zero_metrics() for k in k_list}
        return result

    # 逐位置相关性（二值 + 分级）
    binary_rel: list[int] = []
    graded_rel: list[float] = []
    matched_uids: list[str] = []
    sims = None  # soft 模式下赋值；exact 模式保持 None

    # 是否可用 exact：golden 有 uid 且检索结果也带 uid
    use_exact = bool(golden_uids) and any(c.get("chunk_uid") for c in retrieved)

    if use_exact:
        for c in retrieved:
            uid = c.get("chunk_uid")
            if uid and uid in golden_uids:
                binary_rel.append(1)
                graded_rel.append(1.0)
                matched_uids.append(uid)
            else:
                binary_rel.append(0)
                graded_rel.append(0.0)
        result["mode"] = "exact"
    else:
        # soft：证据语义相似度
        if not golden_evidences:
            result["mode"] = "no_golden_signal"
            result["metrics"] = {k: _zero_metrics() for k in k_list}
            return result
        doc_embs = _embed([c.get("content", "") for c in retrieved])
        ev_embs = _embed(golden_evidences)
        sims = _cosine_matrix(doc_embs, ev_embs)  # (n_docs, n_ev)
        for i in range(sims.shape[0]):
            max_sim = float(sims[i].max()) if sims.shape[1] else 0.0
            binary_rel.append(1 if max_sim >= sim_threshold else 0)
            graded_rel.append(max(0.0, min(1.0, max_sim)))
        result["mode"] = "soft"

    result["metrics"] = compute_retrieval_metrics(graded_rel, k_list, binary=True)
    result["matched_chunk_uids"] = matched_uids

    # golden 级覆盖率（soft 模式用 sims；exact 模式退化为 uid 命中数）
    if golden_evidences and sims is not None and sims.shape[0] and sims.shape[1]:
        covered = 0
        for j in range(sims.shape[1]):
            if float(sims[:, j].max()) >= sim_threshold:
                covered += 1
        result["covered_golden_ratio"] = round(covered / sims.shape[1], 4)
    elif golden_uids:
        covered = len(set(matched_uids))
        result["covered_golden_ratio"] = round(covered / len(golden_uids), 4)

    return result


def _zero_metrics() -> dict:
    return {
        "recall": 0.0,
        "precision": 0.0,
        "hit_rate": 0.0,
        "mrr": 0.0,
        "ndcg": 0.0,
        "f1": 0.0,
    }


def summarize_retrieval(results: list[dict], k_list: list[int] | None = None) -> dict:
    """对多条 case 的检索结果做聚合（仅统计有指标的样本）。"""
    k_list = k_list or [1, 3, 5, 10]
    agg: dict[str, Any] = {"mode_counts": {}, "samples": 0, "per_k": {}}
    mode_counts: dict[str, int] = {}
    per_k: dict[int, dict[str, list[float]]] = {k: {m: [] for m in
        ["recall", "precision", "hit_rate", "mrr", "ndcg", "f1"]} for k in k_list}
    for r in results:
        mode_counts[r.get("mode", "unknown")] = mode_counts.get(r.get("mode", "unknown"), 0) + 1
        for k in k_list:
            mk = (r.get("metrics") or {}).get(k)
            if not mk:
                continue
            for m in per_k[k]:
                per_k[k][m].append(mk[m])
    agg["mode_counts"] = mode_counts
    agg["samples"] = sum(len(per_k[k]["recall"]) for k in k_list) // len(k_list) if k_list else 0
    agg["per_k"] = {k: {m: mean(vals) for m, vals in per_k[k].items()} for k in k_list}
    return agg
