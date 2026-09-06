"""
应用主包 / 导入流程兼容层 / 图编排子模块 / 节点适配层中的 node_audio_to_md 模块，负责承载对应场景的具体实现逻辑。
"""
import os

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import (
    add_done_task,
    add_running_task,
    set_task_result,
    update_task_status,
    TASK_STATUS_FAILED,
)
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.audio_parse_service import audio_parse_to_md
from app.shared.utils.path_util import PROJECT_ROOT
from app.infra.transcription.moss_gateway import moss_gateway
"""
节点作用: node_audio_to_md  将音频转成md,并且保存和存储,同时修改state相关的参数
入参:  [local_file_path/doc_path:str :Path   local_dir:str :Path 默认的存储文件地址(项目/output) ]
出参:  [md_path:str  md_content:str]
步骤:
   1. 日志+进行中的任务记录 add_running_task
   2. 具体的路径校验、音频转写、组织成Markdown与落盘都在 rag 层 service 中处理
   3. 日志+完成的任务记录  add_done_task
"""
@node_log("node_audio_to_md")
def node_audio_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 音频转Markdown (node_audio_to_md)
    """
    # 前置检查：MOSS 转写服务是否启用（受 MOSS_SERVICE_ENABLED 控制）。
    # 提前在节点层拦截，避免底层 HTTP 连接错误（连接被拒绝）冒泡成晦涩堆栈。
    if not moss_gateway.enabled:
        msg = (
            "MOSS 转写服务未启用,无法进行音频转写。请先在 .env 设置 "
            "MOSS_SERVICE_ENABLED=true,并确保已启动 moss 服务进程。"
        )
        logger.error(msg)
        update_task_status(state["task_id"], TASK_STATUS_FAILED, push_queue=True)
        set_task_result(state["task_id"], "error", msg)
        raise RuntimeError(msg)

    # 音频转写需要加载语音大模型，耗时较长，单独记录开始状态便于前端展示进度。
    add_running_task(state["task_id"], "node_audio_to_md")
    # 具体的音频转写、Markdown 组织与落盘都在 rag 层 service 中处理。
    state = audio_parse_to_md(state)
    add_done_task(state["task_id"], "node_audio_to_md")
    return state


if __name__ == "__main__":

    # 单元测试：验证音频转MD全流程（参考 test/audio_to_text.py 使用的测试音频）
    logger.info("===== 开始node_audio_to_md节点单元测试 =====")

    test_audio_name = os.path.join("test", "三体简介.wav")
    audio_path = os.path.join(PROJECT_ROOT, test_audio_name)

    if not os.path.exists(audio_path):
        logger.warning(f"测试文件不存在，跳过本次测试：{audio_path}")
    else:
        test_state = create_default_state(
            task_id="test_audio2md_task_001",
            local_file_path=audio_path,
            local_dir=os.path.join(PROJECT_ROOT, "output")
        )

        result_state = node_audio_to_md(test_state)

        logger.info(f"生成Markdown路径：{result_state.get('md_path')}")
        logger.info(f"Markdown内容预览：\n{str(result_state.get('md_content', ''))[:500]}")

    logger.info("===== 结束node_audio_to_md节点单元测试 =====")
