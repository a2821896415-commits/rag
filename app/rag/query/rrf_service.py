"""
RRF 融合服务模块，负责多路召回结果的融合排序。
"""
from app.shared.runtime.logger import logger, step_log


@step_log("validate_rrf_inputs")
def validate_rrf_inputs(state: dict) -> tuple[list[dict], list[dict]]:
    """
    读取 RRF 融合所需的输入数据。

    Args:
        state: 查询图当前状态。

    Returns:
        tuple[list[dict], list[dict]]: 普通向量召回结果与 HyDE 召回结果。
    """
    return state.get("embedding_chunks", []), state.get("hyde_embedding_chunks", [])


@step_log("reciprocal_rank_fusion")
def reciprocal_rank_fusion(
    param_list: list[tuple[list[dict], float]],
    *,
    k: int = 60,
    top: int = 5,
) -> list[dict]:
    """
    对多路召回结果执行 Reciprocal Rank Fusion 融合。

    Args:
        param_list: 每一路结果及其权重组成的列表。
        k: RRF 的平滑参数。
        top: 最终保留的结果数量。

    Returns:
        list[dict]: 融合后按相关性排序的文档实体列表。
    """
    # score_dict 负责累计同一 chunk 在不同召回路中的 RRF 融合得分。
    score_dict: dict[str, float] = {}
    entity_dict: dict[str, dict] = {}
    for chunks_list, weight in param_list:
        for rank, chunk in enumerate(chunks_list, start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (k + rank)) * weight
            entity_dict.setdefault(chunk_id, chunk)

    document_list = []
    for chunk_id, score in score_dict.items():
        document = entity_dict.get(chunk_id, {}).copy()
        document["score"] = score
        document_list.append(document)
    document_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    final_documents = document_list[:top]
    logger.info(f"RRF融合完成，输入路数:{len(param_list)}，输出条数:{len(final_documents)}")
    return final_documents


@step_log("fuse_retrieval_results")
def fuse_retrieval_results(state: dict) -> list[dict]:
    """
    执行查询链中的 RRF 融合步骤。

    Args:
        state: 查询图当前状态。

    Returns:
        list[dict]: 融合后的文档结果列表。
    """
    # 当前先融合普通向量检索和 HyDE 检索两路结果，后续若扩展更多路可继续加权。
    embedding_chunks, hyde_embedding_chunks = validate_rrf_inputs(state)
    param_list = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0),
    ]
    return reciprocal_rank_fusion(param_list)
