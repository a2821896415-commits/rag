"""
答案生成服务模块，负责 Prompt 组装、模型生成与结果回写。
"""
import re
import time

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.infra.persistence import history_repository
from app.shared.utils.sse_utils import SSEEvent, push_to_session
from app.shared.utils.task_utils import set_task_result

# 不同角色对应不同的回答视角，来源于“听书知识库需求说明”中的角色定义。
ROLE_INSTRUCTIONS = {
    "听众": "你的对话对象是『听众』：请用通俗易懂、亲切的口吻，聚焦书籍介绍、作者信息、听书笔记、推荐理由与常见问题，帮助听众快速了解并挑选适合收听的内容。",
    "内容运营": "你的对话对象是『内容运营』：请侧重梳理书籍资料要点、提炼可复用的推荐话术与文案、归纳常见问题的标准答案，并快速定位知识点，便于运营直接复用。",
    "内容编辑": "你的对话对象是『内容编辑』：请侧重资料的完整性与字段规范性（书名/作者/类别/来源等），校验知识库内容是否准确、一致、无缺失，并指出需要补充或修正之处。",
}


def _role_instruction(role: str) -> str:
    """根据前端传入的角色返回对应的回答视角设定，未知角色默认按听众处理。"""
    return ROLE_INSTRUCTIONS.get(role) or ROLE_INSTRUCTIONS["听众"]


@step_log("try_return_existing_answer")
def try_return_existing_answer(state: dict) -> bool:
    """
    优先返回状态中已经存在的答案，避免重复调用模型。

    Args:
        state: 查询图当前状态，可能已经提前写入 `answer` 字段。

    Returns:
        bool: 如果已经存在可直接返回的答案则返回 True，否则返回 False。
    """
    # 某些分支会提前产出澄清答案，此时直接复用，避免重复调用大模型。
    answer = state.get("answer")
    # ???????????????????????????
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    if not answer:
        return False
    if is_stream:
        # 仍以流式（打字机）逐字推送，但仅保留极小停顿，避免“一个字蹦一秒”的体感。
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.02)
    set_task_result(session_id, "answer", answer)
    return True


@step_log("validate_generation_inputs")
def validate_generation_inputs(state: dict) -> tuple[list[dict], list[str], str, list[dict]]:
    """
    校验答案生成阶段必需的输入是否齐全。

    Args:
        state: 查询图当前状态，需包含重排结果、主体信息和问题文本。

    Returns:
        tuple[list[dict], list[str], str, list[dict]]: 依次返回重排文档、主体列表、查询文本和历史消息。
    """
    history = state.get("history", [])
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    if not reranked_docs or not rewritten_query:
        logger.error("reranked_docs或者rewritten_query为空,无法使用模型进行答案匹配!")
        raise ValueError("reranked_docs或者rewritten_query为空,无法使用模型进行答案匹配!")
    return reranked_docs, item_names, rewritten_query, history


@step_log("build_history_text")
def build_history_text(history_messages: list[dict]) -> str:
    """
    将历史消息拼接为适合 Prompt 使用的纯文本。

    Args:
        history_messages: 历史消息列表，包含角色、文本、改写问题和关联主体等字段。

    Returns:
        str: 拼接后的历史上下文文本。
    """
    if not history_messages:
        return "没有历史聊天记录!"

    lines: list[str] = []
    for msg in history_messages:
        content = msg.get("rewritten_query") if msg.get("role") == "user" else msg.get("text")
        item_names = "、".join(msg.get("item_names", []))
        lines.append(f"角色:{msg.get('role', '')},内容:{content},关联主体: {item_names}")
    return "\n".join(lines)


@step_log("build_answer_prompt")
def build_answer_prompt(
    reranked_docs: list[dict],
    rewritten_query: str,
    item_names: list[str],
    history: list[dict],
    role: str = "",
) -> str:
    """
    构建答案生成 Prompt。

    Args:
        reranked_docs: 重排后的文档列表，作为回答时的主要上下文。
        rewritten_query: 改写后的查询文本。
        item_names: 当前问题关联的主体名称列表。
        history: 最近历史消息列表，用于增强多轮问答上下文。
        role: 当前用户角色（听众/内容运营/内容编辑），用于调整回答视角。

    Returns:
        str: 可直接送入大模型的最终 Prompt 文本。
    """
    context_chunk_list = []
    for number, chunk in enumerate(reranked_docs, start=1):
        # 带上书名/作者/内容类型/来源文件，模型才能区分不同书籍并在答案里给出可追溯的来源。
        source = "网络搜索" if chunk.get("type") == "web" else "知识库"
        meta = (
            f"书名:{chunk.get('book_name') or '未知'}"
            f" 作者:{chunk.get('author') or '未知'}"
            f" 条目名称:{chunk.get('entry_name') or '未知'}"
            f" 内容类型:{chunk.get('content_type') or chunk.get('title') or '未知'}"
            f" 来源文件:{chunk.get('file_title') or '未知'}"
        )
        context_chunk_list.append(
            f"第{number}块: 标题:{chunk.get('title')} {meta} 匹配度得分:{chunk.get('score')} 来源:{source}\n内容:{chunk.get('text')}"
        )
    context_chunk_str = "\n\n".join(context_chunk_list)

    history_text = build_history_text(history)

    item_name_str = "本次关联主体:" + ",".join(item_names) if item_names else "没有关联主体"
    return load_prompt(
        "answer_out",
        context=context_chunk_str,
        history=history_text,
        item_names=item_name_str,
        question=rewritten_query,
        role_instruction=_role_instruction(role),
    )


@step_log("generate_answer")
def generate_answer(state: dict, prompt: str) -> str:
    """
    调用大模型生成答案，并根据模式决定是否推送流式增量。

    Args:
        state: 查询图当前状态，需包含会话 ID 与流式开关。
        prompt: 已组装完成的回答 Prompt。

    Returns:
        str: 大模型生成的最终答案文本。
    """
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    lm_client = llm_provider.chat()
    final_result = ""
    if is_stream:
        for chunk in lm_client.stream(prompt):
            delta_content = chunk.content
            final_result += delta_content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = lm_client.invoke(prompt)
        final_result = response.content
    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    return final_result


@step_log("extract_image_urls")
def extract_image_urls(reranked_docs: list[dict]) -> list[str]:
    """
    从重排文档中提取可用于前端展示的图片链接。

    Args:
        reranked_docs: 重排后的文档列表，可能包含网页直链或 Markdown 图片语法。

    Returns:
        list[str]: 去重后的图片 URL 列表。
    """
    image_urls: list[str] = []
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")
    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text")
        if url and url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)
    return image_urls


@step_log("extract_sources")
def extract_sources(reranked_docs: list[dict]) -> list[dict]:
    """
    从重排结果中提取可追溯的引用信息。

    Args:
        reranked_docs: 重排后的文档列表，可能来自知识库或联网搜索。

    Returns:
        list[dict]: 去重后的来源列表，每条包含书名/作者/内容类型/来源文件等字段。
    """
    sources: list[dict] = []
    seen: set[str] = set()
    for doc in reranked_docs:
        key = str(doc.get("chunk_id") or doc.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": doc.get("title", ""),
                "book_name": doc.get("book_name", ""),
                "author": doc.get("author", ""),
                "entry_name": doc.get("entry_name", ""),
                "content_type": doc.get("content_type", ""),
                "file_title": doc.get("file_title", ""),
                "source": "网络搜索" if doc.get("type") == "web" else "知识库",
                "score": doc.get("score", 0.0),
            }
        )
    logger.info(f"溯源信息提取完成,共{len(sources)}条")
    return sources


@step_log("save_assistant_message")
def save_assistant_message(state: dict) -> None:
    """
    将助手回答写入历史记录。

    Args:
        state: 查询图当前状态，需包含会话 ID、最终答案、改写问题和图片链接等信息。
    """
    history_repository.save_message(
        session_id=state["session_id"],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )


# 检索无可信命中时直接返回的兜底语，避免对空/无关内容调用模型产生幻觉。
NO_RELEVANT_ANSWER = (
    "抱歉，我在知识库中没有找到与该问题相关的内容。"
    "可以换个说法，或提供具体的书名、作者、类别来查询。"
)


@step_log("produce_answer")
def produce_answer(state: dict) -> dict:
    """
    执行答案生成主流程，并将结果写回状态与历史记录。

    Args:
        state: 查询图当前状态。

    Returns:
        dict: 写回最终答案与图片链接后的最新状态。
    """
    # 已经存在可复用答案（如闲聊拦截、澄清反问）时直接收口，避免重复调用模型。
    if try_return_existing_answer(state):
        save_assistant_message(state)
        return state
    # 检索重排后没有任何可信命中，直接给出兜底答案，不再调用大模型生成，
    # 防止空/无关上下文引发的幻觉式长串输出。
    reranked_docs = state.get("reranked_docs", [])
    if not reranked_docs:
        logger.info("重排结果为空,走无相关兜底答案")
        state["answer"] = NO_RELEVANT_ANSWER
        state["sources"] = []
        state["image_urls"] = []
        set_task_result(state["session_id"], "answer", NO_RELEVANT_ANSWER)
        save_assistant_message(state)
        return state
    reranked_docs, item_names, rewritten_query, history = validate_generation_inputs(state)
    prompt = build_answer_prompt(
        reranked_docs,
        rewritten_query,
        item_names,
        history,
        role=state.get("role", ""),
    )
    generate_answer(state, prompt)
    state["image_urls"] = extract_image_urls(reranked_docs)
    # 溯源信息一并返回给前端，满足知识库问答"可解释、可追溯"的要求。
    state["sources"] = extract_sources(reranked_docs)
    save_assistant_message(state)
    return state