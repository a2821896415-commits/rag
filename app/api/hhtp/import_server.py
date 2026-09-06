"""
导入服务 HTTP 入口模块，直接承载导入接口与相关接口业务逻辑。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import List, Dict, Any
from fastapi.responses import FileResponse

# 兼容直接以 `python import_server.py` 方式启动，提前把项目根目录加入模块搜索路径。
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.import_ import ImportStatusResponse, UploadResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state
from app.infra.config import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_done_task, add_running_task,
)


app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/html")
def import_html():
    """
    返回导入演示页面。
    Returns:
        FileResponse: 本地导入演示页面文件响应。
    """
    html_path = PROJECT_ROOT / "app" / "process" / "import_" / "page" / "import.html"
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

# --------------------------
# 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
# --------------------------
def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param local_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param local_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # 2. 初始化LangGraph状态：加载默认状态 + 注入当前任务的核心参数
        init_state = get_default_state()
        init_state["task_id"] = task_id  # 任务ID关联
        init_state["local_dir"] = local_dir  # 任务本地目录
        init_state["local_file_path"] = local_file_path  # 上传文件本地路径

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # 记录每个节点完成的日志，包含任务ID和节点名，方便追踪执行顺序
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)




# --------------------------
# 核心接口：多文件上传接口（不上传 MinIO）
# 支持多文件批量上传，核心流程：接收文件 → 本地保存 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
# --------------------------
from pathlib import Path

@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # 5. 构建上传文件的本地保存绝对路径
        local_file_abs_path: Path = task_local_dir / file.filename

        # 6. 将上传的文件保存到本地临时目录
        with local_file_abs_path.open("wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{local_file_abs_path}")

        # 7. 标记「文件上传」阶段为「已完成」
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return UploadResponse(
        code=200,
        message=f"Files uploaded successfully, total: {len(files)}",
        task_ids=task_ids
    )


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
# 2. 改造接口
@app.get("/status/{task_id}",
         summary="任务状态查询",
         description="根据TaskID查询单个文件的处理进度和全局状态",
         response_model=ImportStatusResponse)  # 绑定模型
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: ImportStatusResponse 格式响应
    """
    # 获取任务各阶段状态
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)

    # 记录日志
    logger.info(f"[{task_id}] 任务状态查询，当前状态：{status}，已完成节点：{done_list}")

    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )


# --------------------------
# 批量任务状态查询接口
# /upload 一次会返回多个 task_id，前端用这个接口一次拿到整体进度
# 访问地址：http://localhost:8000/status/batch?task_ids=id1,id2 （GET请求）
# --------------------------
@app.get("/status/batch", summary="批量任务状态查询", description="一次查询多个 TaskID 的处理进度，用于批量导入")
async def get_batch_task_progress(task_ids: str = ""):
    # 支持逗号分隔的多个任务ID，逐个读取状态后统一汇总。
    id_list = [task_id.strip() for task_id in (task_ids or "").split(",") if task_id.strip()]
    items: List[Dict[str, Any]] = []
    for task_id in id_list:
        items.append(
            {
                "task_id": task_id,
                "status": get_task_status(task_id),
                "done_list": get_done_task_list(task_id),
                "running_list": get_running_task_list(task_id),
            }
        )
    completed = sum(1 for item in items if item["status"] == TASK_STATUS_COMPLETED)
    failed = sum(1 for item in items if item["status"] == TASK_STATUS_FAILED)
    logger.info(f"批量任务状态查询，总数:{len(items)}，已完成:{completed}，失败:{failed}")
    return {
        "code": 200,
        "total": len(items),
        "completed": completed,
        "failed": failed,
        "items": items,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
