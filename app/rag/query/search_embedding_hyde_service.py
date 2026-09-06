"""
HyDE 检索服务模块，负责假设答案生成与 HyDE 检索执行。
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log
from app.infra.llm import llm_provider
from app.rag.query.search_embedding_service import search_chunks, validate_retrieval_state


@step_log("search_embedding_hyde")
def search_embedding_hyde(state: dict) -> list[dict]:
    """
    执行 HyDE 检索的服务总入口。

    Args:
        state: 查询图当前状态，需包含主体列表与改写后的查询文本。

    Returns:
        list[dict]: HyDE 检索得到的文档切块列表。
    """
    item_names, rewritten_query = validate_retrieval_state(state)
    _, chunks = search_chunks_with_hyde(rewritten_query=rewritten_query, item_names=item_names)
    return chunks


@step_log("generate_hyde_answer")
def generate_hyde_answer(rewritten_query: str) -> str:
    """
    基于改写问题生成一段 HyDE 假设答案。

    Args:
        rewritten_query: 改写后的查询文本。

    Returns:
        str: 由模型生成的假设答案文本，用于增强检索召回。
    """
    # 先让模型围绕当前问题生成一段“假设性标准答案”。
    prompt_str = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    messages = [HumanMessage(content=prompt_str)]
    return (llm_provider.chat() | StrOutputParser()).invoke(messages)


@step_log("search_chunks_with_hyde")
def search_chunks_with_hyde(
    *,
    rewritten_query: str,
    item_names: list[str],
    limit: int = 5,
) -> tuple[str, list[dict]]:
    """
    先生成 HyDE 假设答案，再基于拼接查询执行混合检索。

    Args:
        rewritten_query: 改写后的查询文本。
        item_names: 已确认的主体名称列表。
        limit: 最大返回文档数。

    Returns:
        tuple[str, list[dict]]: 依次返回 HyDE 文本和检索到的切块列表。
    """
    # HyDE 的核心就是先补一段假设答案，再扩展检索查询表达。
    hyde_answer = generate_hyde_answer(rewritten_query)
    hybrid_query = f"{rewritten_query},{hyde_answer}"
    chunks = search_chunks(rewritten_query=hybrid_query, item_names=item_names, limit=limit)
    return hyde_answer, chunks
