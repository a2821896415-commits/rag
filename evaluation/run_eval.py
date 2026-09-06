"""
评估引擎编排入口。

用法：
  uv run python evaluation/run_eval.py                 # 全量：理解 + 检索 + 生成评测
  uv run python evaluation/run_eval.py --no-gen        # 仅理解 + 检索（不调用 LLM 生成/裁判）
  uv run python evaluation/run_eval.py --limit 3        # 冒烟：只跑前 3 条 case
  uv run python evaluation/run_eval.py --k 1,3,5,10     # 指定 k
  uv run python evaluation/run_eval.py --rerank         # 检索后套用线上重排
  uv run python evaluation/run_eval.py --out evaluation

数据组织（v2）：dataset.json 顶层 `groups` 为四能力分桶
（comprehension / retrieval / generation / boundary），每条 case 冗余携带 `capability`。
编排对每条 case 先跑【理解能力】评测（驱动真实主体确认链路），
再按是否含 golden_chunks 跑【检索】，按 answer+golden 跑【生成】。

产出：
  evaluation/report.json / evaluation/report.md
"""
import argparse
import json
import sys
from pathlib import Path

# 让脚本可直接以仓库根目录运行
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_retrieval import evaluate_case_retrieval  # noqa: E402
from evaluation.evaluate_generation import evaluate_case_generation  # noqa: E402
from evaluation.evaluate_comprehension import evaluate_case_comprehension  # noqa: E402
from evaluation.report import write_report  # noqa: E402

# 这些期望行为意味着"应正常作答"，才会触发生成评测
_ANSWER_BEHAVIORS = {"answer"}


def flatten_cases(dataset: dict) -> list[dict]:
    """兼容 v2 分桶(groups) 与旧版扁平(cases) 两种结构，统一挂载 capability。"""
    groups = dataset.get("groups")
    if groups:
        cases: list[dict] = []
        for grp in groups:
            cap = grp.get("capability")
            for c in grp.get("cases", []):
                cc = dict(c)
                cc.setdefault("capability", cap)
                cases.append(cc)
        return cases
    return dataset.get("cases", [])


def _should_answer(case: dict) -> bool:
    return (
        case.get("type") == "rag"
        and case.get("expected_behavior") in _ANSWER_BEHAVIORS
        and bool(case.get("golden_chunks"))
    )


def _case_books(case: dict) -> list[str]:
    """提取 case 归属书籍（用于报告按书拆分）。"""
    keys = set()
    for fld in ("expected_item_names", "expect_source_books"):
        for b in (case.get(fld) or []):
            keys.add(b)
    for g in (case.get("golden_chunks") or []):
        if g.get("item_name"):
            keys.add(g["item_name"])
    return sorted(keys)


def main():
    ap = argparse.ArgumentParser(description="RAG 评估引擎（四能力分桶）")
    ap.add_argument("--dataset", default=str(ROOT / "evaluation" / "dataset.json"))
    ap.add_argument("--k", default="1,3,5,10", help="逗号分隔的 k 列表")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 case（0=全部）")
    ap.add_argument("--no-gen", action="store_true", help="跳过生成评测（不调用 LLM）")
    ap.add_argument(
        "--rerank",
        action="store_true",
        help="检索后套用线上重排 rerank_documents（默认关闭，此时测的是原始混合检索排序）",
    )
    ap.add_argument("--out", default=str(ROOT / "evaluation"), help="报告输出目录")
    args = ap.parse_args()

    k_list = [int(x) for x in args.k.split(",") if x.strip()]
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    cases = flatten_cases(dataset)
    if args.limit:
        cases = cases[: args.limit]

    case_results = []
    total = len(cases)
    for idx, case in enumerate(cases, 1):
        cid = case.get("case_id", f"#{idx}")
        cap = case.get("capability", "?")
        print(f"[{idx}/{total}] [{cap}] 评测 {cid} ...", flush=True)

        # 1) 理解能力：驱动真实主体确认链路（不落库历史）
        comprehension = evaluate_case_comprehension(case)

        # 2) 检索能力：golden_chunks 存在才评
        ret = evaluate_case_retrieval(case, k_list=k_list, use_rerank=args.rerank)
        retrieved = ret.pop("retrieved", [])

        # 3) 生成能力：仅正常作答且有 golden 的 case
        generation = None
        if not args.no_gen and _should_answer(case):
            try:
                generation = evaluate_case_generation(case, retrieved)
            except Exception as exc:  # noqa: BLE001
                generation = {"error": str(exc)}

        case_results.append({
            "case_id": cid,
            "capability": cap,
            "type": case.get("type"),
            "expected_behavior": case.get("expected_behavior"),
            "books": _case_books(case),
            "comprehension": comprehension,
            "retrieval": ret,
            "generation": generation,
        })

    json_path, md_path = write_report(case_results, out_dir=args.out)
    print(f"\n完成。报告已写出:\n  - {json_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
