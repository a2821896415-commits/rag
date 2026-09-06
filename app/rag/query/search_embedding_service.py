"""
向量检索服务模块，负责普通 Embedding 检索的参数校验与执行。
"""
import json

from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.vectorstore import milvus_gateway


def _parse_category(raw) -> list:
    """把 chunk 的 category 字段统一解析成列表（写入时为 JSON 字符串，读取时还原）。"""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


RETRIEVAL_DEFAULT_LIMIT = 5
RETRIEVAL_RANKER_WEIGHTS = (0.9, 0.1)


@step_log("search_embedding")
def search_embedding(state: dict) -> list[dict]:
    """
    执行普通向量检索的服务总入口。

    Args:
        state: 查询图当前状态，需包含主体列表与改写后的查询文本。

    Returns:
        list[dict]: 普通向量检索得到的文档切块列表。
    """
    item_names, rewritten_query = validate_retrieval_state(state)
    return search_chunks(rewritten_query=rewritten_query, item_names=item_names)


@step_log("validate_retrieval_state")
def validate_retrieval_state(state: dict) -> tuple[list[str], str]:
    """
    校验检索前置条件是否满足。

    Args:
        state: 查询图当前状态，需包含主体列表与改写后的查询文本。

    Returns:
        tuple[list[str], str]: 依次返回主体名称列表与改写查询。
    """
    item_names = state.get("item_names")
    rewritten_query = state.get("rewritten_query")
    if not item_names or not rewritten_query:
        logger.error("item_names或rewritten_query不存在,无法继续业务!")
        raise ValueError("item_names或rewritten_query不存在,无法继续业务!")
    return item_names, rewritten_query


@step_log("build_item_name_expr")
def build_item_name_expr(item_names: list[str]) -> str:
    """
    构建 Milvus 过滤表达式，用于限定检索主体范围。

    Args:
        item_names: 已确认的主体名称列表。

    Returns:
        str: 适用于 Milvus 查询的表达式字符串。
    """
    return f"item_name in {item_names}"


@step_log("normalize_retrieved_chunk")
def normalize_retrieved_chunk(chunk: dict) -> dict:
    """
    将 Milvus 检索结果归一化为查询链内部统一使用的文档结构。

    Args:
        chunk: Milvus 返回的原始切块结果。

    Returns:
        dict: 标准化后的检索文档。
    """
    entity = chunk.get("entity", chunk)
    return {
        "chunk_id": chunk.get("id") or entity.get("chunk_id"),
        "chunk_uid": entity.get("chunk_uid", ""),
        "item_name": entity.get("item_name", ""),
        "title": entity.get("title"),
        "parent_title": entity.get("parent_title"),
        "part": entity.get("part"),
        "file_title": entity.get("file_title"),
        "content": entity.get("content", ""),
        # 以下为导入链广播的书籍元数据，用于答案溯源（书名/作者/类别/内容类型）。
        "book_name": entity.get("book_name", ""),
        "author": entity.get("author", ""),
        "entry_name": entity.get("entry_name", ""),
        "category": _parse_category(entity.get("category")),
        "content_type": entity.get("content_type", ""),
        "score": chunk.get("distance", 0.0),
        "type": "milvus",
        "url": None,
    }


@step_log("search_chunks")
def search_chunks(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
) -> list[dict]:
    """
    基于改写问题执行一次混合向量检索。

    Args:
        rewritten_query: 用于检索的改写后问题。
        item_names: 已确认的主体名称列表，用于过滤知识范围。
        limit: 最大返回文档数。

    Returns:
        list[dict]: 检索得到的切块结果列表。
    """
    # 先把改写后的问题编码成 dense/sparse 两类向量，和导入链保持同口径。
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0]
    # 基于主体过滤条件构造混合检索请求，限定召回范围只落在当前产品内。
    reqs = milvus_gateway.create_requests(
        dense_vector,
        sparse_vector,
        expr=build_item_name_expr(item_names),
        limit=limit,
    )
    # 混合检索后会把 Milvus 原始结果统一整理成查询链内部标准结构。
    resp = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunks_collection,
        reqs=reqs,
        ranker_weights=RETRIEVAL_RANKER_WEIGHTS,
        norm_score=True,
        limit=limit,
        output_fields=[
            "chunk_id", "item_name", "content", "title", "parent_title", "part", "file_title",
            # 元数据广播字段，用于答案溯源与结构化展示
            "book_name", "author", "entry_name", "category", "content_type", "chunk_uid",
        ],
    )
    return [normalize_retrieved_chunk(chunk) for chunk in (resp[0] if resp else [])]
