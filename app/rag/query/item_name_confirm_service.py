"""
商品确认服务模块，负责问题改写、主体抽取、候选匹配与确认结果回写。
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.persistence import history_repository
from app.infra.vectorstore import milvus_gateway

# 识别为纯闲聊时直接返回的友好兜底语，不进入检索与生成。
CHITCHAT_FALLBACK = (
    "你好！我是听书知识库助手，可以帮你查询某本书的简介、作者、听书笔记、"
    "推荐资料等。告诉我书名或作者名，我就能为你检索～"
)

QUERY_HISTORY_LIMIT = 10
ITEM_NAME_CONFIRM_THRESHOLD = 0.55
ITEM_NAME_CANDIDATE_THRESHOLD = 0.40
ITEM_NAME_OPTIONS_TOPK = 2


@step_log("validate_query_identity")
def validate_query_identity(state: dict) -> tuple[str, str]:
    """
    校验查询状态中是否包含主体确认所需的核心字段。

    Args:
        state: 查询图当前状态，理论上至少应包含 `original_query` 与 `session_id`。

    Returns:
        tuple[str, str]: 依次返回原始问题与会话 ID。
    """
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query or not session_id:
        logger.error("session_id和original_query不能为空")
        raise ValueError("session_id和original_query不能为空")
    return original_query, session_id


@step_log("load_history")
def load_history(session_id: str) -> list[dict]:
    """
    读取当前会话最近的历史消息。

    Args:
        session_id: 当前会话 ID。

    Returns:
        list[dict]: 最近若干条历史消息，用于问题改写和主体识别。
    """
    return history_repository.list_recent(session_id, limit=QUERY_HISTORY_LIMIT)


@step_log("build_history_text")
def build_history_text(history_messages: list[dict]) -> str:
    """
    将历史消息拼接为适合 Prompt 使用的纯文本。

    Args:
        history_messages: 历史消息列表，包含角色、文本、改写问题和关联主体等字段。

    Returns:
        str: 拼接后的历史上下文文本。
    """
    lines: list[str] = []
    for msg in history_messages:
        content = msg.get("rewritten_query") if msg.get("role") == "user" else msg.get("text")
        item_names = "、".join(msg.get("item_names", []))
        lines.append(f"角色:{msg.get('role', '')},内容:{content},关联主体: {item_names}")
    return "\n".join(lines)


@step_log("rewrite_query_and_extract_item_names")
def rewrite_query_and_extract_item_names(history_messages: list[dict], original_query: str) -> dict:
    """
    在一次模型调用中同时完成问题改写与主体抽取。

    Args:
        history_messages: 最近历史消息列表，用于帮助模型理解上下文。
        original_query: 用户当前输入的原始问题。

    Returns:
        dict: 至少包含 `rewritten_query` 与 `item_names` 两个字段的结果字典。
    """
    client = llm_provider.chat(json_mode=True)
    prompt = load_prompt(
        "rewritten_query_and_itemnames",
        history_text=build_history_text(history_messages),
        query=original_query,
    )
    messages = [
        SystemMessage(content="你是一个听书知识库助手，擅长理解用户的听书意图，并准确提取书名、作者、类别或听书场景等关键信息。"),
        HumanMessage(content=prompt),
    ]
    result = (client | JsonOutputParser()).invoke(messages)
    if "rewritten_query" not in result:
        logger.warning(f"模型重写问题失败,给rewritten_query赋予原始问题:{original_query}")
        result["rewritten_query"] = original_query
    if "item_names" not in result:
        logger.warning("模型识别书籍失败,给item_names赋予空列表")
        result["item_names"] = []
    if "is_chitchat" not in result:
        result["is_chitchat"] = False
    return result


@step_log("search_item_name_candidates")
def search_item_name_candidates(item_names: list[str]) -> dict[str, list[dict]]:
    """
    基于候选主体名称到 Milvus 中检索相近主体。

    Args:
        item_names: 模型抽取出来的主体名称列表。

    Returns:
        dict[str, list[dict]]: 以原始主体名为键、相似主体候选列表为值的映射结果。
    """
    vector_dict: dict[str, list[dict]] = {}
    item_name_vectors = llm_provider.embed_documents(item_names)
    for index, item_name in enumerate(item_names):
        dense_vector = item_name_vectors["dense"][index]
        sparse_vector = item_name_vectors["sparse"][index]
        reqs = milvus_gateway.create_requests(dense_vector, sparse_vector)
        response = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.5, 0.5),
            norm_score=True,
            output_fields=["item_name"],
        )
        current_item_name_list: list[dict] = []
        for item in (response[0] if response else []):
            current_item_name_list.append(
                {
                    "item_name": item.get("entity", {}).get("item_name", ""),
                    "score": item.get("distance", 0),
                }
            )
        vector_dict[item_name] = current_item_name_list
    return vector_dict


@step_log("select_item_names")
def select_item_names(vector_dict: dict[str, list[dict]]) -> dict:
    """
    按阈值策略从主体候选中选出确认项或待澄清项。

    Args:
        vector_dict: 主体候选结果字典，键为原主体名，值为相似主体及分数列表。

    Returns:
        dict: 包含 `confirmed_item_name_list` 与 `options_item_name_list` 的判定结果。
    """
    confirmed_item_name_list: list[str] = []
    options_item_name_list: list[str] = []
    for _, item_name_list in vector_dict.items():
        item_name_list.sort(key=lambda x: x["score"], reverse=True)
        high_list = [item for item in item_name_list if item["score"] >= ITEM_NAME_CONFIRM_THRESHOLD]
        low_list = [
            item
            for item in item_name_list
            if ITEM_NAME_CANDIDATE_THRESHOLD <= item["score"] < ITEM_NAME_CONFIRM_THRESHOLD
        ]
        if high_list:
            confirmed_item_name_list.append(high_list[0]["item_name"])
            continue
        if low_list:
            options_item_name_list.extend([item["item_name"] for item in low_list[:ITEM_NAME_OPTIONS_TOPK]])
    return {
        "confirmed_item_name_list": confirmed_item_name_list,
        "options_item_name_list": options_item_name_list,
    }


@step_log("apply_item_name_result")
def apply_item_name_result(state: dict, final_result: dict, rewritten_query: str) -> None:
    """
    将主体确认结果写回查询状态，并在需要时生成澄清回复。

    Args:
        state: 查询图当前状态，会被原地修改。
        final_result: 主体确认后的结果字典，包含确认主体和候选主体。
        rewritten_query: 问题改写后的查询文本。
    """
    confirmed_item_name_list = final_result.get("confirmed_item_name_list", [])
    options_item_name_list = final_result.get("options_item_name_list", [])
    if confirmed_item_name_list:
        state["item_names"] = confirmed_item_name_list
        state["rewritten_query"] = rewritten_query
        if "answer" in state:
            del state["answer"]
        return
    if options_item_name_list:
        option_name_str = "、".join(options_item_name_list)
        state["answer"] = f"您是想问以下哪本书：{option_name_str}？请告诉我具体书名或作者。"
        state["rewritten_query"] = rewritten_query
        state["item_names"] = []
        return
    state["answer"] = "抱歉，知识库里没有找到相关书籍，请提供书名或作者名以便我为您查询。"
    state["rewritten_query"] = rewritten_query
    state["item_names"] = []


@step_log("save_user_message")
def save_user_message(state: dict) -> None:
    """
    将用户消息及其改写结果写入历史记录。

    Args:
        state: 查询图当前状态，需包含会话 ID、原始问题、改写问题与主体信息。
    """
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
    )


@step_log("confirm_item_name")
def confirm_item_name(state: dict) -> dict:
    """
    执行主体确认主流程。

    Args:
        state: 查询图当前状态，需提供原始问题与会话 ID。

    Returns:
        dict: 写回主体确认结果后的最新状态。
    """
    # 先取出并校验本轮查询的核心身份信息：问题文本和会话 ID。
    original_query, session_id = validate_query_identity(state)
    # 带上最近历史消息做问题改写，尽量提升多轮问答的上下文理解能力。
    history_messages = load_history(session_id)
    # 回填到 state，答案生成节点才能拿到真实历史（之前该字段全程为空，多轮等于没有）。
    state["history"] = history_messages
    # 一次模型调用同时完成“问题改写 + 主体抽取 + 闲聊识别”，减少链路长度。
    llm_result = rewrite_query_and_extract_item_names(history_messages, original_query)
    item_names = llm_result["item_names"]
    rewritten_query = llm_result["rewritten_query"]
    is_chitchat = bool(llm_result.get("is_chitchat"))
    # 纯闲聊且未抽到任何主体：直接友好收口，跳过检索/重排/生成，避免无谓的多轮模型调用拖慢响应。
    if is_chitchat and not item_names:
        logger.info(f"识别为闲聊/无书籍意图,直接收口:{original_query}")
        state["answer"] = CHITCHAT_FALLBACK
        state["item_names"] = []
        state["rewritten_query"] = original_query
        save_user_message(state)
        return state
    final_result = {}
    # 只有抽到了主体候选，才继续去主体向量库里做确认和筛选。
    if item_names:
        final_result = select_item_names(search_item_name_candidates(item_names))
    # 把主体确认结果和改写后的问题统一写回 state，供后续检索节点使用。
    apply_item_name_result(state, final_result, rewritten_query)
    save_user_message(state)
    return state
