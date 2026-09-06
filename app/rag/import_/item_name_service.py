"""
主体识别服务模块，负责从导入文档中识别主体名称并写入主体索引。
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.rag.import_.config import (
    ITEM_NAME_CONTEXT_CHUNK_K,
    ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)
from app.shared.utils.escape_milvus_string_utils import escape_milvus_string


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state: dict) -> tuple[list[dict], str]:
    chunks = state.get("chunks", [])
    file_title = state.get("file_title")
    if not chunks:
        logger.error("chunks没有内容,无法继续业务!")
        raise ValueError("chunks没有内容,无法继续业务!")
    if not file_title:
        logger.warning("file_title为空给与默认值处理!")
        file_title = "default_title"
        state["file_title"] = file_title
    return chunks, file_title


@step_log("build_document_context")
def build_document_context(chunks: list[dict]) -> str:
    current_chunks = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    chunk_str_list: list[str] = []
    for index, item in enumerate(current_chunks, start=1):
        chunk_str_list.append(f"切片:{index},标题:{item['title']},内容:{item['content']}")
    chunk_str = "\n".join(chunk_str_list)
    return chunk_str[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]


def build_item_name_record(item_name: str, alias: str, alias_type: str, file_title: str) -> dict:
    """
    构造一条主体索引记录。

    主体索引里 `item_name` 恒为书名，`alias` 才是用户可能说出口的说法（作者、类别、文件名等）。
    查询阶段拿别名文本做向量匹配，命中后返回 `item_name` 去过滤 chunks 集合，
    两者必须解耦，否则用"刘慈欣"去过滤 chunks 会一条都命中不了。
    """
    return {
        "item_name": item_name,
        "alias": alias,
        "alias_type": alias_type,
        "file_title": file_title,
    }


@step_log("recognize_item_name")
def recognize_item_name(chunks: list[dict], file_title: str) -> list[dict]:
    # 书名/作者/类别等元数据已由 split_service 广播到每段，这里直接复用，不再重复解析。
    # 文档级 item_name 统一取书名：同一本书的所有段落共用一个值，检索时才能一次过滤出整本书，
    # 也避免不同书籍的同名段落（内容正文/核心看点）在主体索引里互相覆盖。
    if not chunks:
        logger.warning("chunks没有内容,主体名回退为文件标题!")
        return [build_item_name_record(file_title, file_title, "file", file_title)]

    doc_file_title = file_title
    doc_item_name = ""
    for chunk in chunks:
        doc_file_title = chunk.get("file_title") or doc_file_title
        # 书名缺失时退化为文件标题，保证每段仍然有可用的文档级主体。
        if not doc_item_name:
            doc_item_name = (chunk.get("book_name") or "").strip()
    doc_item_name = doc_item_name or doc_file_title

    for chunk in chunks:
        chunk["item_name"] = doc_item_name

    # 汇总该书的所有别名来源：作者、条目名称、类别标签、内容类型、文件名。
    authors: list[str] = []
    entry_names: list[str] = []
    categories: list[str] = []
    content_types: list[str] = []
    for chunk in chunks:
        for source, value in (
            (authors, chunk.get("author")),
            (entry_names, chunk.get("entry_name")),
            (content_types, chunk.get("content_type")),
        ):
            value = (value or "").strip()
            if value and value not in source:
                source.append(value)
        for tag in (chunk.get("category") or []):
            tag = (tag or "").strip()
            if tag and tag not in categories:
                categories.append(tag)

    records: list[dict] = [build_item_name_record(doc_item_name, doc_item_name, "book", doc_file_title)]
    for author in authors:
        records.append(build_item_name_record(doc_item_name, author, "author", doc_file_title))
    for tag in categories:
        records.append(build_item_name_record(doc_item_name, tag, "category", doc_file_title))
    for entry_name in entry_names:
        # 条目名与书名相同时已由 book 记录覆盖，避免写入重复别名。
        if entry_name == doc_item_name:
            continue
        records.append(build_item_name_record(doc_item_name, entry_name, "entry_name", doc_file_title))
    for content_type in content_types:
        records.append(build_item_name_record(doc_item_name, content_type, "content_type", doc_file_title))
    if doc_file_title and doc_file_title != doc_item_name:
        records.append(build_item_name_record(doc_item_name, doc_file_title, "file", doc_file_title))

    logger.info(f"主体识别完成,文档级主体:{doc_item_name},共整理{len(records)}条主体记录")
    return records


@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str) -> list[dict]:
    for chunk in chunks:
        chunk["item_name"] = item_name
    return chunks


@step_log("embed_item_name")
def embed_item_name(item_name_records: list[dict]) -> tuple[list[list[float]], list[dict[int, float]]]:
    # 向量由别名文本生成：用户说的是"刘慈欣""科幻"，主体索引里要匹配的也是这些说法。
    alias_list = [record["alias"] for record in item_name_records]
    result = llm_provider.embed_documents(alias_list)
    return result["dense"], result["sparse"]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection() -> None:
    milvus_client = milvus_gateway.client()
    collection_name = milvus_gateway.item_name_collection
    if milvus_client.has_collection(collection_name=collection_name):
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)
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


@step_log("upsert_item_name")
def upsert_item_name(item_name_records: list[dict], file_title: str, dense_vector: list, sparse_vector: list) -> None:
    if not item_name_records:
        logger.warning("没有需要写入主体索引的记录,跳过写入!")
        return

    milvus_client = milvus_gateway.client()
    prepare_item_name_collection()
    # 按文件标题清理：同一文件重复导入时先删旧数据，既保证覆盖旧版本，
    # 又不会像按 item_name 删除那样把其它书籍的同名段落记录一起删掉。
    safe_file_title = escape_milvus_string(file_title)
    milvus_client.delete(
        collection_name=milvus_gateway.item_name_collection,
        filter=f"file_title == '{safe_file_title}'",
    )

    rows: list[dict] = []
    for index, record in enumerate(item_name_records):
        rows.append(
            {
                "file_title": record["file_title"],
                "item_name": record["item_name"],
                "alias": record["alias"],
                "alias_type": record["alias_type"],
                "dense_vector": dense_vector[index],
                "sparse_vector": sparse_vector[index],
            }
        )
    milvus_client.insert(
        collection_name=milvus_gateway.item_name_collection,
        data=rows,
    )
    logger.info(f"主体索引写入完成,共{len(rows)}条,文件:{file_title}")


@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: dict) -> dict:
    chunks, file_title = validate_chunks_and_title(state)
    # 段落自带元数据，逐段确认文档级主体，并整理出要写入主体索引的别名记录。
    item_name_records = recognize_item_name(chunks, file_title)
    # state 里保留文档级主体（书名），供后续节点与日志使用。
    state["item_name"] = item_name_records[0]["item_name"] if item_name_records else file_title
    # 主体别名本身也会生成向量，便于查询阶段做主体确认。
    dense_vector, sparse_vector = embed_item_name(item_name_records)
    upsert_item_name(item_name_records, file_title, dense_vector, sparse_vector)
    return state
