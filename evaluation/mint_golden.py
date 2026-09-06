# -*- coding: utf-8 -*-
"""辅助脚本：向线上知识库查询候选 golden 锚点，供全新编写 retrieval/generation case 时策展真实 chunk_uid。

用法：
  uv run python evaluation/mint_golden.py "《三体》的故事起点" 三体
  uv run python evaluation/mint_golden.py "适合睡前听的书"        # 不限定书，跨库检索

输出：top-k 候选的 chunk_uid / item_name / content_type / 内容片段，可直接抄入 dataset.json 的 golden_chunks。
注意：本脚本只读不写，仅作策展辅助；绝不自行编造 uid。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.query.search_embedding_service import search_chunks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="用于检索的自然语言问题")
    ap.add_argument("item_name", nargs="?", default=None, help="可选：限定主体书名，不传则跨库")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    chunks = search_chunks(rewritten_query=args.query, item_names=[args.item_name] if args.item_name else [], limit=args.k)
    if not chunks:
        print("（无候选，可能 item_name 不在库或 Milvus 不可达）")
        return
    for i, c in enumerate(chunks, 1):
        print(f"--- 候选 {i} ---")
        print("chunk_uid   :", c.get("chunk_uid"))
        print("item_name   :", c.get("item_name"))
        print("content_type:", c.get("content_type"))
        print("score       :", round(c.get("score", 0.0), 4))
        print("content     :", (c.get("content") or "")[:120].replace("\n", " "))
        print()


if __name__ == "__main__":
    main()
