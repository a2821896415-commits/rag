"""
向量化服务模块，负责为文档切块批量生成稠密与稀疏向量。
"""
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import EMBEDDING_BATCH_SIZE


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
        logger.error("chunks为空,无法继续业务处理!")
        raise ValueError("chunks为空,无法继续业务处理!")
    return chunks


@step_log("embed_chunks")
def embed_chunks(chunks: list[dict], *, step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    chunks_vector: list[dict] = []
    total = len(chunks)
    for index in range(0, total, step):
        try:
            step_chunks = chunks[index:index + step]
            vector_str_list = []
            for item in step_chunks:
                item_name = item.get("title")
                content = item.get("content", "")
                vector_str_list.append(f"主体:{item_name},内容:{content}" if item_name else content)
            result = llm_provider.embed_documents(vector_str_list)
            for i, chunk in enumerate(step_chunks):
                chunk_new = chunk.copy()
                chunk_new["dense_vector"] = result["dense"][i]
                chunk_new["sparse_vector"] = result["sparse"][i]
                chunks_vector.append(chunk_new)
        except Exception as exc:
            logger.warning(f"index={index}步骤,发生错误,跳过,继续生成向量!!,错误信息:{str(exc)}")
            continue
    return chunks_vector


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: dict) -> dict:
    # 先确认 chunks 存在，再批量写回 dense/sparse 向量字段。
    state["chunks"] = embed_chunks(require_chunks(state))
    return state
