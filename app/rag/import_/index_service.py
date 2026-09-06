"""
入库服务模块，负责创建 Milvus 集合并写入切块数据。
"""
import json

from pymilvus import DataType

from app.shared.runtime.logger import logger, step_log
from app.infra.vectorstore import milvus_gateway
from app.rag.import_.config import (
    MILVUS_CHUNK_CONTENT_MAX_LENGTH,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """
    校验导入状态中是否已经生成切块结果。

    Args:
        state: 导入图当前状态。

    Returns:
        list[dict]: 已通过校验的切块列表。
    """
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空,无法继续业务!!")
        raise ValueError("chunks为空,无法继续业务!!")
    return chunks


@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.chunks_collection
    # 兼容旧集合：已存在但缺少元数据字段时，销毁后按新 schema 重建，
    # 否则书名/作者/类别/内容类型等字段无法落库，溯源与"按标签推荐"都会失效。
    if milvus_client.has_collection(collection_name=collection_name):
        existing = milvus_client.describe_collection(collection_name)
        field_names = {f.get("name") for f in existing.get("fields", [])}
        required = {"book_name", "author", "entry_name", "category", "content_type", "chunk_uid"}
        if not required.issubset(field_names):
            logger.warning(f"chunks 集合缺少元数据字段 {required - field_names}，重建以兼容新 schema")
            milvus_client.drop_collection(collection_name)
        else:
            return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH)
    # 导入链广播的书籍元数据，作为一等字段可靠存储，支撑溯源与按标签过滤/推荐。
    schema.add_field(field_name="book_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="entry_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    # 稳定唯一标识，供评估 golden 与 Milvus 精确锚定（见 P0-2）；不依赖 auto_id，重入库仍稳定。
    schema.add_field(field_name="chunk_uid", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        index_name="dense_vector_index",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)


@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str) -> None:
    # 按文件标题清理：同一文件重复导入时覆盖旧数据，
    # 也不会像按 item_name 删除那样误删其它书籍里同名的段落。
    safe_file_title = escape_milvus_string(file_title)
    milvus_gateway.client().delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"file_title=='{safe_file_title}'",
    )


@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]) -> None:
    # 显式构造写入行，并把 category 列表序列化为 JSON 字符串（VARCHAR 兼容各 Milvus 版本，
    # 避免 list 类型依赖动态字段而静默丢失）。
    rows: list[dict] = []
    for chunk in chunks:
        raw_cat = chunk.get("category") or []
        cat_str = json.dumps(list(raw_cat), ensure_ascii=False) if isinstance(raw_cat, (list, tuple)) else str(raw_cat)
        rows.append(
            {
                "file_title": chunk.get("file_title", ""),
                "item_name": chunk.get("item_name", ""),
                "title": chunk.get("title", ""),
                "parent_title": chunk.get("parent_title", ""),
                "part": chunk.get("part", 0),
                "content": chunk.get("content", ""),
                "book_name": chunk.get("book_name", ""),
                "author": chunk.get("author", ""),
                "entry_name": chunk.get("entry_name", ""),
                "content_type": chunk.get("content_type", ""),
                "chunk_uid": chunk.get("chunk_uid", ""),
                "category": cat_str,
                "dense_vector": chunk.get("dense_vector"),
                "sparse_vector": chunk.get("sparse_vector"),
            }
        )
    result = milvus_gateway.client().insert(
        collection_name=milvus_gateway.chunks_collection,
        data=rows,
    )
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")


@step_log("index_chunks")
def index_chunks(state: dict) -> dict:
    # 先校验切片存在，避免把空数据写入向量库。
    chunks = require_chunks(state)
    # 集合不存在时先自动创建，保证首次导入也能直接跑通。
    prepare_chunks_collection()
    file_title = state.get("file_title", "")
    # 同一文件重复导入时先删旧数据，保持当前导入结果覆盖旧版本。
    if file_title:
        remove_old_chunks(file_title)
    insert_chunks(chunks)
    return state
