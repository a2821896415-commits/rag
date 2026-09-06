"""
联网检索服务模块，负责通过 DashScope MCP 执行 WebSearch 并解析结果。
"""
import asyncio
import json

from agents.mcp.server import MCPServerStreamableHttp
from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.runtime.logger import logger, step_log

DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key


@step_log("validate_web_search_inputs")
def validate_web_search_inputs(state: dict) -> str:
    """
    校验联网检索所需的查询文本。

    Args:
        state: 查询图当前状态，需至少包含 `rewritten_query`。

    Returns:
        str: 已校验通过的改写查询文本。
    """
    rewritten_query = state.get("rewritten_query")
    if not rewritten_query:
        logger.error("rewritten_query不能为空!")
        raise ValueError("rewritten_query不能为空!")
    return rewritten_query


async def search_web_documents_async(rewritten_query: str, count: int = 5):
    """
    通过 DashScope MCP 异步执行一次联网搜索。

    Args:
        rewritten_query: 改写后的查询文本。
        count: 搜索返回的最大结果数。
    """
    # 1. 链接mcpserver服务
    mcp_server = MCPServerStreamableHttp(
        name="search_mcp",  # 随便写
        client_session_timeout_seconds=300,
        params={
            "url": DASHSCOPE_BASE_URL_STREAM_ABLE_HTTP,  #
            "headers": {"Authorization": DASHSCOPE_API_KEY},
            "timeout": 300,
            "sse_read_timeout": 300
        })
    try:
        # 2. 建立 MCP 连接并拉取工具清单，便于排查远端服务是否可用。
        await mcp_server.connect()
        tool_list = await mcp_server.list_tools()
        logger.info(f"工具列表:{tool_list}")
        # 3. 直接调用百炼 WebSearch 工具，返回原始 MCP 响应结果。
        return await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": rewritten_query,
                "count": count,
            },
        )
    finally:
        await mcp_server.cleanup()


@step_log("search_web_documents")
def search_web_documents(state: dict, count: int = 10) -> list[dict]:
    """
    执行联网检索并将 MCP 结果解析为页面列表。

    Args:
        state: 查询图当前状态。
        count: 搜索返回的最大结果数。

    Returns:
        list[dict]: 联网搜索得到的页面结果列表。
    """
    # 先校验改写问题，再把 MCP 原始结果解析成网页列表。
    rewritten_query = validate_web_search_inputs(state)
    mcp_result = asyncio.run(search_web_documents_async(rewritten_query, count=count))
    text_dict = json.loads(mcp_result.content[0].text)
    return text_dict.get("pages", [])
