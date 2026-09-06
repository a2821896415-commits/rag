"""
查询服务 HTTP 入口模块，直接承载查询接口与相关接口业务逻辑。
"""
import json
import sys
import uuid
from mimetypes import guess_type
from pathlib import Path

# 兼容直接以 `python query_server.py` 方式启动，提前把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.query import HistoryItem, HistoryResponse, QueryRequest, QueryResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config import settings
from app.infra.persistence.history_repository import history_repository
from app.infra.vectorstore import milvus_gateway
from app.process.query.agent.main_graph import query_app as query_graph_app
from app.process.query.agent.state import create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)


app = FastAPI(
    title=settings.query_app_name,
    description="企业化 RAG 查询服务，负责问答、SSE 输出与历史记录查询。",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def new_session_id() -> str:
    """
    生成新的会话 ID。

    Returns:
        str: 基于 UUID 生成的唯一会话标识。
    """
    return str(uuid.uuid4())


def invoke_query(session_id: str, query: str, is_stream: bool, role: str = "") -> dict:
    """
    调用查询主图并维护统一的任务状态。

    Args:
        session_id: 当前会话 ID。
        query: 用户原始问题。
        is_stream: 是否为流式模式。
        role: 当前用户角色（听众/内容运营/内容编辑），用于调整回答视角。

    Returns:
        dict: 查询图执行后的最终状态。
    """
    clear_task(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    initial_state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
        role=role,
    )
    try:
        state = query_graph_app.invoke(initial_state)
    except Exception as exc:
        # 需求要求单模块异常不影响整体服务，这里兜底成失败状态并给出可解释的提示。
        logger.exception(f"查询主图执行失败，session_id={session_id}, error={exc}")
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        return {
            "answer": "抱歉，本次查询处理失败，请稍后重试或换个问法。",
            "sources": [],
            "image_urls": [],
        }
    update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    return state


def run_stream_query_background(session_id: str, query: str, role: str = "") -> None:
    """
    在后台执行一次流式查询任务。

    Args:
        session_id: 当前流式查询对应的会话 ID。
        query: 用户原始问题。
        role: 当前用户角色（听众/内容运营/内容编辑）。
    """
    try:
        state = invoke_query(session_id=session_id, query=query, is_stream=True, role=role)
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": get_task_result(session_id, "answer") or state.get("answer", ""),
                "status": "completed",
                "image_urls": state.get("image_urls", []),
                "sources": state.get("sources", []),
            },
        )
    except Exception as exc:
        logger.exception(f"流式查询执行失败，session_id={session_id}, error={exc}")
        update_task_status(session_id, TASK_STATUS_FAILED, True)
        push_to_session(
            session_id,
            SSEEvent.ERROR,
            {"message": "查询执行失败，请检查日志或稍后重试。"},
        )


def start_stream_query(
    background_tasks: BackgroundTasks,
    query: str,
    session_id: str | None = None,
    role: str = "",
) -> QueryResponse:
    """
    启动一条流式查询任务。

    Args:
        background_tasks: FastAPI 后台任务对象。
        query: 用户原始问题。
        session_id: 可选会话 ID；为空时自动生成。
        role: 当前用户角色（听众/内容运营/内容编辑）。

    Returns:
        QueryResponse: 返回处理中提示与当前会话 ID。
    """
    final_session_id = session_id or new_session_id()
    create_sse_queue(final_session_id)
    background_tasks.add_task(
        run_stream_query_background,
        final_session_id,
        query,
        role,
    )
    return QueryResponse(message="结果正在处理中...", session_id=final_session_id)


def execute_query(query: str, session_id: str | None = None, role: str = "") -> QueryResponse:
    """
    以非流式方式执行查询。

    Args:
        query: 用户原始问题。
        session_id: 可选会话 ID；为空时自动生成。
        role: 当前用户角色（听众/内容运营/内容编辑）。

    Returns:
        QueryResponse: 包含最终答案和已完成节点的响应对象。
    """
    final_session_id = session_id or new_session_id()
    state = invoke_query(session_id=final_session_id, query=query, is_stream=False, role=role)
    answer = get_task_result(final_session_id, "answer") or state.get("answer", "")
    return QueryResponse(
        message="处理完成！",
        session_id=final_session_id,
        answer=answer,
        done_list=get_done_task_list(final_session_id),
        image_urls=state.get("image_urls", []),
        sources=state.get("sources", []),
    )


def build_history_response(session_id: str, limit: int = 10) -> HistoryResponse:
    """
    查询并组装指定会话的历史记录。

    Args:
        session_id: 目标会话 ID。
        limit: 返回记录条数上限。

    Returns:
        HistoryResponse: 组装后的历史消息集合。
    """
    records = history_repository.list_recent(session_id, limit=limit)
    items = [
        HistoryItem(
            _id=str(record.get("_id")) if record.get("_id") is not None else "",
            session_id=record.get("session_id", ""),
            role=record.get("role", ""),
            text=record.get("text", ""),
            rewritten_query=record.get("rewritten_query", ""),
            item_names=record.get("item_names", []),
            image_urls=record.get("image_urls", []),
            ts=record.get("ts"),
        )
        for record in records
    ]
    return HistoryResponse(session_id=session_id, items=items)


def clear_query_history(session_id: str) -> dict:
    """
    清空指定会话的历史记录。

    Args:
        session_id: 目标会话 ID。

    Returns:
        dict: 删除结果说明。
    """
    delete_count = history_repository.clear_session(session_id)
    return {
        "message": f"删除:{session_id}会话对应的聊天记录成功!!",
        "deleted_count": delete_count,
    }


# 书籍列表扫描时最多读取的切块行数，覆盖常规知识库规模即可。
BOOK_METADATA_SCAN_LIMIT = 10000


def parse_book_category(raw) -> list:
    """
    把 Milvus 中 category 字段统一解析为标签列表（入库时为 JSON 字符串）。

    Args:
        raw: Milvus 返回的原始 category 值。

    Returns:
        list: 解析后的标签列表。
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def ensure_collection_loaded(client, collection_name: str) -> None:
    """
    集合未加载到内存时先加载，保证标量查询可用。

    Args:
        client: Milvus 客户端实例。
        collection_name: 目标集合名称。
    """
    try:
        if "Loaded" not in str(client.get_load_state(collection_name=collection_name)):
            client.load_collection(collection_name=collection_name)
    except Exception:
        # 加载状态判断失败时仍尝试加载一次；加载本身失败则交由后续查询兜底。
        try:
            client.load_collection(collection_name=collection_name)
        except Exception as exc:
            logger.warning(f"集合加载失败，书籍列表查询可能受限: {exc}")


def collect_books(limit: int = BOOK_METADATA_SCAN_LIMIT) -> dict:
    """
    扫描切块集合，按书名聚合出书籍清单与全局标签，供书籍速览页动态渲染。

    Args:
        limit: 标量查询最大返回行数。

    Returns:
        dict: 包含 ok、书籍列表与标签集合的结果。
    """
    client = milvus_gateway.client()
    collection_name = milvus_gateway.chunks_collection
    if client is None:
        logger.error("列出书籍失败：Milvus 客户端不可用")
        return {"ok": False, "reason": "milvus_unavailable", "books": [], "tags": []}
    try:
        if not client.has_collection(collection_name=collection_name):
            return {"ok": True, "books": [], "tags": []}
        ensure_collection_loaded(client, collection_name)
        rows = client.query(
            collection_name=collection_name,
            output_fields=["book_name", "author", "category", "content_type"],
            limit=limit,
        )
    except Exception as exc:
        logger.exception(f"列出书籍失败: {exc}")
        return {"ok": False, "reason": "query_failed", "books": [], "tags": []}

    books: dict[str, dict] = {}
    tags: set[str] = set()
    for row in rows or []:
        book_name = str(row.get("book_name") or "").strip()
        if not book_name:
            continue
        book = books.setdefault(
            book_name,
            {"author": "", "categories": [], "content_types": [], "_cats": set()},
        )
        author = str(row.get("author") or "").strip()
        if author and not book["author"]:
            book["author"] = author
        for tag in parse_book_category(row.get("category")):
            tags.add(tag)
            if tag not in book["_cats"]:
                book["_cats"].add(tag)
                book["categories"].append(tag)
        content_type = str(row.get("content_type") or "").strip()
        if content_type and content_type not in book["content_types"]:
            book["content_types"].append(content_type)

    book_list = [
        {
            "book_name": name,
            "author": book["author"],
            "categories": book["categories"],
            "content_types": book["content_types"],
        }
        for name, book in books.items()
    ]
    return {"ok": True, "books": book_list, "tags": sorted(tags)}


@app.get("/health")
def health():
    """
    返回查询服务健康检查结果。

    Returns:
        dict: 包含当前服务名称、环境与模块信息。
    """
    return {
        "ok": True,
        "app": settings.query_app_name,
        "env": settings.app_env,
        "module": "query",
    }


@app.get("/")
def index():
    """
    返回查询服务首页导航信息。

    Returns:
        dict: 包含页面地址、查询接口、流式接口与历史接口导航信息。
    """
    return {
        "message": "Enterprise RAG Query Service",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "routes": {
            "html": "/html",
            "console": "/console",
            "query": "/query",
            "stream": "/stream/{session_id}",
            "history": "/history/{session_id}",
        },
    }


@app.get("/html")
def query_html():
    """
    返回查询演示页面。

    Returns:
        FileResponse: 本地聊天演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/console")
def console_html():
    """
    返回听书知识库全屏控制台页面（问答 / 导入 / 书籍速览）。

    Returns:
        FileResponse: 本地控制台页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "web" / "console.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/books")
def list_books():
    """
    返回知识库已收录书籍的聚合信息与全局标签。

    Returns:
        dict: 书籍列表与标签集合，导入新书后会自动纳入其中。
    """
    return collect_books()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    统一处理普通问答与流式问答请求。

    Args:
        request: 查询请求体，包含问题文本、是否流式输出和可选会话 ID。
        background_tasks: FastAPI 后台任务对象，流式模式下用于异步启动查询任务。

    Returns:
        QueryResponse: 非流式时返回完整答案；流式时返回会话 ID 与处理中提示。
    """
    if request.is_stream:
        return start_stream_query(
            background_tasks=background_tasks,
            query=request.query,
            session_id=request.session_id,
            role=request.role,
        )
    return execute_query(query=request.query, session_id=request.session_id, role=request.role)


@app.get("/stream/{session_id}")
async def stream_query_result(session_id: str, request: Request):
    """
    建立指定会话的 SSE 结果流。

    Args:
        session_id: 目标会话 ID。
        request: 当前 HTTP 请求对象，用于感知客户端断开连接。

    Returns:
        StreamingResponse: 基于 SSE 协议的流式响应对象。
    """
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str, limit: int = 10):
    """
    查询会话历史记录。

    Args:
        session_id: 目标会话 ID。
        limit: 返回记录条数上限。

    Returns:
        HistoryResponse: 当前会话最近的历史消息集合。
    """
    return build_history_response(session_id=session_id, limit=limit)


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """
    清空指定会话的历史记录。

    Args:
        session_id: 目标会话 ID。

    Returns:
        dict: 删除结果说明。
    """
    return clear_query_history(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
