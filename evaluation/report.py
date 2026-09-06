"""
评估报告聚合与落盘。

输入：run_eval 产出的逐 case 结果列表，每条形如
{
  "case_id", "type", "expected_behavior",
  "retrieval": {...evaluate_case_retrieval 结果...},
  "generation": {...evaluate_case_generation 结果 or None...},
}

输出：
- evaluation/report.json：完整结构化结果
- evaluation/report.md：人类可读摘要
"""
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import mean

_K_LIST = [1, 3, 5, 10]
_RETR_METRICS = ["recall", "precision", "hit_rate", "mrr", "ndcg", "f1"]

# 这些期望行为意味着"不应正常作答/检索"
_NO_RETRIEVE_BEHAVIORS = {"refuse", "refuse_book_content", "clarify", "reject", "negative"}

# 四能力维度（与 dataset.json meta.capability_legend 对齐）
_CAPABILITY_ORDER = ["comprehension", "retrieval", "generation", "boundary"]


def _comprehension_block(case_results: list[dict]) -> dict:
    """聚合理解能力指标：意图/实体/行为准确率与综合分。"""
    intent, ent, beh, score = [], [], [], []
    samples = 0
    for r in case_results:
        c = r.get("comprehension")
        if not c or c.get("error"):
            continue
        samples += 1
        if c.get("intent_accuracy") is not None:
            intent.append(c["intent_accuracy"])
        if c.get("entity_accuracy") is not None:
            ent.append(c["entity_accuracy"])
        if c.get("behavior_accuracy") is not None:
            beh.append(c["behavior_accuracy"])
        if c.get("comprehension_score") is not None:
            score.append(c["comprehension_score"])
    return {
        "samples": samples,
        "intent_accuracy": mean(intent),
        "entity_accuracy": mean(ent),
        "behavior_accuracy": mean(beh),
        "comprehension_score": mean(score),
    }


def _retrieval_block(case_results: list[dict]) -> dict:
    """聚合检索指标：整体 + 各 k + 模式分布 + golden 覆盖率。"""
    per_k: dict[int, dict[str, list[float]]] = {
        k: {m: [] for m in _RETR_METRICS} for k in _K_LIST
    }
    covered: list[float] = []
    mode_counts: dict[str, int] = {}
    valid = 0
    for r in case_results:
        ret = r.get("retrieval") or {}
        mode_counts[ret.get("mode", "unknown")] = mode_counts.get(ret.get("mode", "unknown"), 0) + 1
        # 仅对"含 golden_chunks"的 case 计入覆盖率：no_golden / no_golden_signal
        # 代表无 golden 可比对（其 covered_golden_ratio 为占位 0.0），不应拉低均值。
        if ret.get("mode") not in ("no_golden", "no_golden_signal") and ret.get(
            "covered_golden_ratio"
        ) is not None:
            covered.append(ret["covered_golden_ratio"])
        metrics = ret.get("metrics") or {}
        for k in _K_LIST:
            mk = metrics.get(k)
            if not mk:
                continue
            for m in _RETR_METRICS:
                if mk.get(m) is not None:
                    per_k[k][m].append(mk[m])
        if metrics:
            valid += 1
    return {
        "samples": valid,
        "mode_counts": mode_counts,
        "covered_golden_ratio": mean(covered),
        "per_k": {
            k: {m: mean(per_k[k][m]) for m in _RETR_METRICS} for k in _K_LIST
        },
    }


def _generation_block(case_results: list[dict]) -> dict:
    """聚合生成指标。"""
    faith, rel, cov = [], [], []
    samples = 0
    for r in case_results:
        gen = r.get("generation")
        if not gen:
            continue
        samples += 1
        if gen.get("faithfulness") is not None:
            faith.append(gen["faithfulness"])
        if gen.get("relevance") is not None:
            rel.append(gen["relevance"])
        if gen.get("fact_coverage") is not None:
            cov.append(gen["fact_coverage"])
    return {
        "samples": samples,
        "faithfulness": mean(faith),
        "relevance": mean(rel),
        "fact_coverage": mean(cov),
    }


def _split_key(case: dict) -> list[str]:
    """返回该 case 归属的维度键（按书籍）。优先用编排层预存的 books 列表。"""
    books = case.get("books")
    if books:
        return list(books)
    keys = set()
    for fld in ("expected_item_names", "expect_source_books"):
        for b in (case.get(fld) or []):
            keys.add(b)
    for g in (case.get("golden_chunks") or []):
        if g.get("item_name"):
            keys.add(g["item_name"])
    return sorted(keys) or ["<未知>"]


def aggregate_results(case_results: list[dict]) -> dict:
    """聚合逐 case 结果为整体 / 按能力 / 按类型 / 按书籍四层视角。"""
    overall = {
        "retrieval": _retrieval_block(case_results),
        "generation": _generation_block(case_results),
        "comprehension": _comprehension_block(case_results),
        "total_cases": len(case_results),
    }

    by_capability: dict[str, list[dict]] = {}
    by_type: dict[str, list[dict]] = {}
    by_book: dict[str, list[dict]] = {}
    for r in case_results:
        by_capability.setdefault(r.get("capability", "unknown"), []).append(r)
        by_type.setdefault(r.get("type", "unknown"), []).append(r)
        for bk in _split_key(r):
            by_book.setdefault(bk, []).append(r)

    result = {"overall": overall}

    def _sub_block(sub: list[dict]) -> dict:
        return {
            "retrieval": _retrieval_block(sub),
            "generation": _generation_block(sub),
            "comprehension": _comprehension_block(sub),
            "cases": len(sub),
        }

    # 按能力切片（主维度）：固定四能力顺序，缺失能力也给出空桶
    result["by_capability"] = {
        cap: _sub_block(by_capability.get(cap, [])) for cap in _CAPABILITY_ORDER
    }
    # 兼容：保留实际出现的能力
    for cap, sub in by_capability.items():
        if cap not in result["by_capability"]:
            result["by_capability"][cap] = _sub_block(sub)

    result["by_type"] = {t: _sub_block(sub) for t, sub in by_type.items()}
    result["by_book"] = {bk: _sub_block(sub) for bk, sub in by_book.items()}

    # 失败 / 风险标记
    failures: list[dict] = []
    for r in case_results:
        cid = r.get("case_id")
        ret = r.get("retrieval") or {}
        if ret.get("error"):
            failures.append({"case_id": cid, "kind": "retrieval_error", "detail": ret["error"]})
            continue
        # 不该检索却检索到 → 知识泄漏风险
        if (r.get("expected_behavior") in _NO_RETRIEVE_BEHAVIORS) and ret.get("retrieved_count", 0) > 0:
            failures.append({
                "case_id": cid,
                "kind": "unexpected_retrieval",
                "detail": f"期望不检索，却返回 {ret.get('retrieved_count')} 个切块",
            })
    result["failures"] = failures
    return result


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in rows)
    return head + "\n" + sep + "\n" + body


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# RAG 评估摘要报告\n")
    ov = report["overall"]
    lines.append(f"总 case 数：**{ov['total_cases']}**\n")

    lines.append("## 检索指标（整体）")
    retr = ov["retrieval"]
    lines.append(f"- 有效样本: {retr['samples']} | golden 覆盖率: {retr['covered_golden_ratio']}")
    lines.append(f"- 模式分布: {retr['mode_counts']}\n")
    headers = ["k", "recall", "precision", "hit_rate", "mrr", "ndcg", "f1"]
    rows = []
    for k in _K_LIST:
        mk = retr["per_k"].get(k, {})
        rows.append([k] + [mk.get(m, "") for m in _RETR_METRICS])
    lines.append(_md_table(headers, rows))
    lines.append("")

    gen = ov["generation"]
    lines.append("## 生成指标（整体）")
    lines.append(f"- 有效样本: {gen['samples']}")
    lines.append(f"- faithfulness: **{gen['faithfulness']}** | relevance: **{gen['relevance']}** | fact_coverage: **{gen['fact_coverage']}**\n")

    lines.append("## 按类型")
    rows = []
    for t, blk in report["by_type"].items():
        r = blk["retrieval"]["per_k"].get(5, {})
        g = blk["generation"]
        rows.append([t, blk["cases"], r.get("recall", ""), r.get("ndcg", ""), g.get("faithfulness", ""), g.get("relevance", "")])
    lines.append(_md_table(["type", "cases", "recall@5", "ndcg@5", "faithful", "relevant"], rows))
    lines.append("")

    lines.append("## 按书籍（检索 recall@5 / ndcg@5）")
    rows = []
    for bk, blk in sorted(report["by_book"].items()):
        r = blk["retrieval"]["per_k"].get(5, {})
        rows.append([bk, blk["cases"], r.get("recall", ""), r.get("ndcg", ""), blk["retrieval"]["mode_counts"]])
    lines.append(_md_table(["book", "cases", "recall@5", "ndcg@5", "modes"], rows))
    lines.append("")

    fails = report.get("failures") or []
    lines.append(f"## 风险/失败标记（{len(fails)}）")
    if fails:
        for f in fails[:30]:
            lines.append(f"- `{f['case_id']}` [{f['kind']}] {f['detail']}")
    else:
        lines.append("- 无")
    return "\n".join(lines)


def write_report(case_results: list[dict], out_dir: str | Path = "evaluation") -> tuple[Path, Path]:
    """聚合并写出 report.json 与 report.md，返回两者路径。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = aggregate_results(case_results)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
