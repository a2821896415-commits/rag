"""
dataset.json 一致性校验脚本（评估计划 B）。

双层校验：
- Tier1（永远执行，纯 JSON，无需数据库）：
  * 结构完整性：每个 case 的必需键、枚举值、类型是否正确
  * 内部一致性：recommend 的 expected_item_names 与 expect_source_books 对齐、
    golden_chunks / hard_facts 字段齐全、多轮 case 带 history、case_id 唯一等
  * 黄金集与库的弱关联：golden item_name 是否属于 meta.kb_books
- Tier2（best-effort，需要 Milvus 可达）：
  * golden_chunks.content_type 是否真实存在于 chunks 集合
  * expected_item_names / expect_source_books 是否真实存在于 item_name 集合
  * （P0-2 之后）golden_chunks.chunk_uid 是否可解析

退出码：错误数（>0 即失败）；仅 warning 不影响退出码，便于 CI 区分。

运行：
  uv run python evaluation/lint_dataset.py
  uv run python evaluation/lint_dataset.py --strict   # 把 warning 也当作失败
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "dataset.json"

BEHAVIOR_ENUM = {"answer", "refuse", "refuse_book_content", "clarify", "chitchat"}
ROLE_ENUM = {"听众", "内容运营", "内容编辑"}
TYPE_ENUM = {"rag", "intent", "negative"}
FACET_ENUM = {"category", "scene", "author"}
SCENARIO_ENUM = {
    "book_recommend", "book_detail", "content_retrieval",
    "listening_note", "chitchat", "out_of_kb",
}
# 四能力分桶（v2 结构顶层 groups[].capability 的合法取值）
CAPABILITY_ENUM = {"comprehension", "retrieval", "generation", "boundary"}


class LintResult:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def err(self, case_id, msg):
        self.errors.append(f"[ERROR] {case_id}: {msg}")

    def warn(self, case_id, msg):
        self.warnings.append(f"[WARN]  {case_id}: {msg}")

    @property
    def ok(self):
        return not self.errors


# ----------------------------- 数据结构归一（兼容 v2 groups 与旧版 cases） -----------------------------
def flatten_cases(data: dict, res: LintResult) -> list[dict]:
    """把 v2 分桶(groups) 或 旧版扁平(cases) 统一展开为 case 列表，并继承 capability。

    同时做分桶层校验：groups 必须存在且 capability 合法、case 数量平衡可提示。
    """
    groups = data.get("groups")
    if groups is None:
        return data.get("cases", [])

    if not isinstance(groups, list) or not groups:
        res.err("__groups__", "顶层 groups 缺失或为空，无法识别四能力分桶")
        return []

    flat: list[dict] = []
    for gi, grp in enumerate(groups):
        if not isinstance(grp, dict):
            res.err(f"group[{gi}]", "group 必须是对象")
            continue
        cap = grp.get("capability")
        if cap not in CAPABILITY_ENUM:
            res.err(f"group[{gi}]", f"capability 非法: {cap}（合法: {sorted(CAPABILITY_ENUM)}）")
        cs = grp.get("cases", [])
        if not isinstance(cs, list) or not cs:
            res.warn(f"group[{gi}]", f"capability={cap} 下无 case")
        for c in cs:
            cc = dict(c)
            cc.setdefault("capability", cap)
            flat.append(cc)
    return flat


# ----------------------------- Tier1：结构与内部一致性 -----------------------------
def lint_structure(data: dict, cases: list[dict], res: LintResult) -> set:
    meta = data.get("meta", {})
    kb_books = set(meta.get("kb_books", []))
    seen_ids = set()

    for case in cases:
        cid = case.get("case_id", "<no-id>")
        if cid in seen_ids:
            res.err(cid, "case_id 重复")
        seen_ids.add(cid)

        # 基础键与枚举
        for k in ("case_id", "type", "scenario", "query", "expected_intent",
                  "expected_item_names", "difficulty", "expected_behavior", "role"):
            if k not in case:
                res.err(cid, f"缺少必需键 {k}")

        if case.get("type") not in TYPE_ENUM:
            res.err(cid, f"type 非法: {case.get('type')}")
        if case.get("scenario") not in SCENARIO_ENUM:
            res.err(cid, f"scenario 非法: {case.get('scenario')}")
        if case.get("expected_behavior") not in BEHAVIOR_ENUM:
            res.err(cid, f"expected_behavior 非法: {case.get('expected_behavior')}")
        if case.get("role") not in ROLE_ENUM:
            res.err(cid, f"role 非法: {case.get('role')}")
        if case.get("capability") not in CAPABILITY_ENUM:
            res.err(cid, f"capability 非法: {case.get('capability')}（合法: {sorted(CAPABILITY_ENUM)}）")

        # 理解能力分桶：必须能驱动真实确认链路（意图/实体/行为）
        if case.get("capability") == "comprehension":
            if not isinstance(case.get("expected_intent"), dict) or "is_chitchat" not in case.get("expected_intent", {}):
                res.err(cid, "comprehension case 的 expected_intent.is_chitchat 必填")
            if case.get("expected_behavior") is None:
                res.err(cid, "comprehension case 的 expected_behavior 必填（answer/clarify/refuse/chitchat）")

        intent = case.get("expected_intent", {})
        if not isinstance(intent, dict) or "is_chitchat" not in intent or "has_book_intent" not in intent:
            res.err(cid, "expected_intent 需含 is_chitchat/has_book_intent")

        # 多轮必须带 history
        if cid.startswith("mt-") or case.get("history") is not None:
            hist = case.get("history", [])
            if not isinstance(hist, list) or not hist:
                res.err(cid, "多轮 case 需提供非空 history")
            else:
                for h in hist:
                    if not isinstance(h, dict) or "role" not in h or "content" not in h:
                        res.err(cid, "history 条目需含 role/content")

        # 按类型细化
        lint_by_type(case, cid, kb_books, res)

    return kb_books


def lint_by_type(case: dict, cid: str, kb_books: set, res: LintResult):
    ctype = case.get("type")
    scenario = case.get("scenario")

    if ctype == "rag":
        golden = case.get("golden_chunks")
        if not isinstance(golden, list) or not golden:
            res.err(cid, "rag case 需有非空 golden_chunks")
        else:
            for g in golden:
                for k in ("item_name", "content_type", "evidence", "required"):
                    if k not in g:
                        res.err(cid, f"golden_chunks 缺键 {k}")
                # 黄金 item_name 应属于知识库书籍（弱校验）
                if g.get("item_name") and g["item_name"] not in kb_books:
                    res.warn(cid, f"golden item_name 不在 meta.kb_books: {g['item_name']}")

        for k in ("hard_facts", "answer_keypoints", "forbidden", "expect_source_books"):
            if k not in case:
                res.err(cid, f"rag case 缺键 {k}")

        if scenario == "book_recommend":
            if case.get("query_facet") not in FACET_ENUM:
                res.err(cid, f"book_recommend 需 query_facet∈{FACET_ENUM}，实际 {case.get('query_facet')}")
            exp_books = case.get("expect_source_books", [])
            item_names = case.get("expected_item_names", [])
            if not item_names:
                res.err(cid, "book_recommend 的 expected_item_names（推荐书）不应为空")
            if set(item_names) != set(exp_books):
                res.err(cid, f"book_recommend expected_item_names 应与 expect_source_books 一致: "
                             f"{item_names} vs {exp_books}")
        else:
            # book_detail/content_retrieval/listening_note 应指明具体书
            if not case.get("expected_item_names"):
                res.warn(cid, "expected_item_names 为空（非推荐场景通常应有具体书）")

    elif ctype == "intent":
        if case.get("expected_behavior") not in ("chitchat", "clarify"):
            res.warn(cid, f"intent case 的 expected_behavior 通常为 chitchat/clarify，实际 {case.get('expected_behavior')}")

    elif ctype == "negative":
        if case.get("expected_behavior") not in ("refuse", "refuse_book_content"):
            res.err(cid, f"negative case expected_behavior 应为 refuse/refuse_book_content，实际 {case.get('expected_behavior')}")
        if case.get("scenario") != "out_of_kb":
            res.warn(cid, f"negative case 的 scenario 建议为 out_of_kb，实际 {case.get('scenario')}")


# ----------------------------- Tier2：在线比对 Milvus -----------------------------
def lint_against_milvus(cases: list[dict], res: LintResult):
    # 把项目根加入 sys.path，使脚本在 evaluation/ 下也能 `import app`
    sys.path.insert(0, str(HERE.parent))
    try:
        from app.infra.vectorstore import milvus_gateway
    except Exception as exc:  # noqa: BLE001
        res.warn("__db__", f"无法导入 app/vectorstore，跳过在线校验: {exc}")
        return

    try:
        client = milvus_gateway.client()
    except Exception as exc:  # noqa: BLE001
        res.warn("__db__", f"Milvus 不可用，跳过在线校验: {exc}")
        return

    # 兼容性取某集合某字段的所有取值。
    # 注意：Milvus query 不接受 expr=""，需用主键恒真表达式；两个集合主键名不同。
    def distinct_values(collection: str, field: str, id_field: str) -> set:
        try:
            rows = client.query(
                collection,
                filter_expr=f"{id_field} >= 0",
                output_fields=[field],
                limit=10000,
            )
            return {r.get(field) for r in rows if r.get(field)}
        except Exception as exc:  # noqa: BLE001
            res.warn("__db__", f"读取 {collection}.{field} 失败: {exc}")
            return set()

    # 1) 取 chunks 集合 distinct content_type（主键 chunk_id）
    db_content_types = distinct_values(milvus_gateway.chunks_collection, "content_type", "chunk_id")
    # 2) 取 item_name 集合 distinct item_name（主键 pk）
    db_item_names = distinct_values(milvus_gateway.item_name_collection, "item_name", "pk")
    # 3) 取 chunks 集合 distinct chunk_uid（仅当库已含该字段；旧集合未重入库时为空）
    db_chunk_uids = set()
    try:
        fields = {f.get("name") for f in milvus_gateway.client().describe_collection(milvus_gateway.chunks_collection).get("fields", [])}
        if "chunk_uid" in fields:
            db_chunk_uids = distinct_values(milvus_gateway.chunks_collection, "chunk_uid", "chunk_id")
        else:
            res.warn("__db__", "chunks 集合尚未含 chunk_uid 字段（旧集合未用新 schema 重入库），跳过 chunk_uid 在线校验")
    except Exception as exc:  # noqa: BLE001
        res.warn("__db__", f"读取 chunks 集合字段信息失败: {exc}")

    if not db_item_names:
        res.warn("__db__", "item_name 集合无数据，跳过逐条 item_name 校验")
    if not db_content_types:
        res.warn("__db__", "chunks 集合无 content_type 数据，跳过逐条 content_type 校验")

    seen_item_warn = set()
    missing_books = set()
    missing_content_types = set()
    golden_uids: set[str] = set()

    for case in cases:
        cid = case.get("case_id", "<no-id>")
        # content_type 校验（仅 rag 有 golden_chunks）
        for g in case.get("golden_chunks", []):
            ct = g.get("content_type")
            if ct and db_content_types and ct not in db_content_types:
                if ct not in missing_content_types:
                    missing_content_types.add(ct)
                    res.warn(cid, f"golden content_type 不在库中: {ct}（库含 {sorted(db_content_types)}）")
            # chunk_uid 格式校验（P0-2 已落地）
            uid = g.get("chunk_uid")
            if uid is not None:
                if not (isinstance(uid, str) and __import__("re").fullmatch(r"[0-9a-f]{16}", uid)):
                    res.warn(cid, f"golden chunk_uid 格式异常: {uid}")
                golden_uids.add(uid)
        # item_name 校验：仅对"应命中知识库"的 case 检查（rag + 正常作答）。
        # refuse/clarify/negative 的书本就不在库（测试拒答/追问），不应误报。
        if case.get("type") == "rag" and case.get("expected_behavior") == "answer" and db_item_names:
            for name in set(case.get("expected_item_names", [])) | set(case.get("expect_source_books", [])):
                if name and name not in db_item_names and (cid, name) not in seen_item_warn:
                    seen_item_warn.add((cid, name))
                    missing_books.add(name)
                    res.warn(cid, f"item_name 不在主体库: {name}")

    # chunk_uid 在线校验：若库已含 chunk_uid 字段，则 golden 锚点必须能在库中找到
    if db_chunk_uids:
        for uid in golden_uids:
            if uid not in db_chunk_uids:
                res.warn("__db__", f"golden chunk_uid 不在 Milvus 中（可能该书未按新 schema 重入库）: {uid}")

    # 汇总：数据集中引用但库里缺失的书，便于一次性发现"库未入库"的漂移
    if missing_books:
        res.warn("__summary__",
                 f"数据集引用但向量库缺失的书籍（共 {len(missing_books)} 本）: {sorted(missing_books)}")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="把 warning 也当作失败")
    args = ap.parse_args()

    data = json.loads(SRC.read_text(encoding="utf-8"))
    res = LintResult()

    cases = flatten_cases(data, res)
    lint_structure(data, cases, res)
    lint_against_milvus(cases, res)

    print(f"cases: {len(cases)}")
    print(f"errors  : {len(res.errors)}")
    print(f"warnings: {len(res.warnings)}")
    for line in res.errors:
        print(line)
    for line in res.warnings:
        print(line)

    if args.strict and res.warnings:
        sys.exit(len(res.errors) + len(res.warnings))
    sys.exit(len(res.errors))


if __name__ == "__main__":
    main()
