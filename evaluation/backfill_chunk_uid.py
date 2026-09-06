"""
回填 golden_chunks.chunk_uid 脚本（对应评估计划 P0-2）。

做了什么：
1. 扫描仓库内所有 chunks.json（output/**、source/**），这些是 split 阶段落盘的切片。
2. 用与 split_service.compute_chunk_uid 完全相同的公式，为每个 chunk 算出稳定 chunk_uid。
3. 对 dataset.json 里每个 golden_chunks 条目，按
   (item_name, content_type, evidence⊂content) 匹配到具体 chunk，写入其 chunk_uid。
4. 写回 dataset.json（先备份为 dataset.json.pre_chunkuid.bak）。

注意：
- 公式只依赖切分阶段即确定的字段（file_title/parent_title/part/content），
  因此未来用新代码重新入库时，Milvus 里的 chunk_uid 会与这里的回填值一致，
  评估即可做精确 recall@k / mrr / ndcg。
- 若某本书尚未导入（环境里只有部分书），其 golden 匹配不到 chunk，会列在 unmatched 里，
  待该书入库并重跑本脚本即可补齐。

运行：uv run python evaluation/backfill_chunk_uid.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "dataset.json"
BAK = HERE / "dataset.json.pre_chunkuid.bak"

# 把项目根加入 sys.path 以便复用 split_service.compute_chunk_uid
sys.path.insert(0, str(ROOT))
from app.rag.import_.split_service import compute_chunk_uid  # noqa: E402

CHUNKS_GLOBS = ["output", "source"]


def load_all_chunks() -> list[dict]:
    chunks: list[dict] = []
    seen_uids = set()
    for base in CHUNKS_GLOBS:
        for p in (ROOT / base).rglob("chunks.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] 读取失败 {p}: {exc}")
                continue
            for c in data:
                if not isinstance(c, dict) or not c.get("content"):
                    continue
                c["_chunk_uid"] = compute_chunk_uid(c)
                if c["_chunk_uid"] in seen_uids:
                    continue  # 多次导入产生的重复切片，跳过
                seen_uids.add(c["_chunk_uid"])
                chunks.append(c)
    return chunks


def _clean(s: str) -> str:
    # golden 的 evidence 常含省略号（……/…）或人工截断，清洗后再做子串匹配
    return (s or "").replace("……", "").replace("…", "").strip()


def find_chunk(golden: dict, chunks_by_ct: dict, all_chunks: list[dict]) -> dict | None:
    ct = golden.get("content_type")
    ev = _clean(golden.get("evidence", ""))
    item = golden.get("item_name", "")
    if not ev:
        return None
    # 候选池优先级：同书同类型 > 任意书同类型 > 同书全部类型 > 全部（跨类型兜底）
    # 真实 markdown 未必有 golden 写的 content_type（如"用户评论摘要"），故需跨类型兜底。
    pool_ct = chunks_by_ct.get(ct, [])
    same_book_ct = [c for c in pool_ct if c.get("book_name") == item]
    same_book_all = [c for c in all_chunks if c.get("book_name") == item]
    pools: list[list[dict]] = []
    if same_book_ct:
        pools.append(same_book_ct)
    if pool_ct:
        pools.append(pool_ct)
    if same_book_all:
        pools.append(same_book_all)
    pools.append(all_chunks)

    for pool in pools:
        for k in (len(ev), 30, 15):
            if k <= 0:
                continue
            needle = ev[:k]
            for c in pool:
                if needle in c.get("content", ""):
                    return c

    # 兜底：证据文本对不上时，按 (书名, 内容类型) 唯一定位。
    # 每本书每个 content_type 恰好一个切片，故可确定性锚定，
    # 避免这类 golden 拿不到 uid 而在评测时退化成 soft 语义匹配（精度更低）。
    if item and ct:
        cands = [c for c in chunks_by_ct.get(ct, []) if c.get("book_name") == item]
        if len(cands) == 1:
            return cands[0]

    return None


def main() -> None:
    shutil.copy(SRC, BAK)
    data = json.loads(SRC.read_text(encoding="utf-8"))
    chunks = load_all_chunks()
    chunks_by_ct: dict[str, list[dict]] = {}
    for c in chunks:
        chunks_by_ct.setdefault(c.get("content_type", ""), []).append(c)

    total = matched = unmatched = 0
    unmatched_examples: list[str] = []

    for case in data.get("cases", []):
        cid = case.get("case_id", "<no-id>")
        for g in case.get("golden_chunks", []):
            if not isinstance(g, dict):
                continue
            total += 1
            hit = find_chunk(g, chunks_by_ct, chunks)
            if hit:
                g["chunk_uid"] = hit["_chunk_uid"]
                matched += 1
            else:
                unmatched += 1
                if len(unmatched_examples) < 20:
                    unmatched_examples.append(
                        f"{cid} | item={g.get('item_name')} | ct={g.get('content_type')} | ev={g.get('evidence','')[:24]}…"
                    )

    SRC.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"扫描到 chunk 数（去重后）: {len(chunks)}")
    print(f"golden_chunks 总数: {total}")
    print(f"已匹配 chunk_uid: {matched}")
    print(f"未匹配: {unmatched}")
    if unmatched:
        print("未匹配样例（通常为该书尚未入库，入库后重跑本脚本即可补齐）:")
        for ex in unmatched_examples:
            print("  -", ex)
    print(f"备份 -> {BAK}")


if __name__ == "__main__":
    main()
