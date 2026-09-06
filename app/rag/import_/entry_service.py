"""
导入入口服务模块，负责识别输入文件类型并初始化导入状态。
"""
from pathlib import Path
from app.shared.runtime.logger import logger, step_log
from app.process.import_.agent.state import FileType

@step_log("resolve_input_file")
def resolve_input_file(state: dict) -> dict:
    """
    识别导入文件类型并初始化后续节点所需的状态字段。

    Args:
        state: 导入图当前状态，需至少包含 `local_file_path`。

    Returns:
        dict: 补充了文件路径、文件标题及开关字段后的最新状态。
    """
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.warning("节点:node_entry,获取文件输入地址,发现地址为空!直接跳转到END节点")
        return state

    # 后缀统一转小写再判断，避免 .HTML / .PDF 这类大小写差异被误判为不支持。
    suffix = Path(local_file_path).suffix.lower()

    # 识别文件类型，写入单一 file_type 字段（导入图路由的唯一定策依据）。
    if suffix == ".md":
        state["md_path"] = local_file_path
        state["file_type"] = FileType.MD
    elif suffix == ".pdf":
        state["pdf_path"] = local_file_path
        state["file_type"] = FileType.PDF
    elif suffix in (".html", ".htm"):
        state["html_path"] = local_file_path
        state["file_type"] = FileType.HTML
    elif suffix in (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".amr", ".opus"):
        # 音频类输入：交给 MOSS 转写服务（独立 venv，依赖 transformers 5.x）转成 Markdown
        state["file_type"] = FileType.AUDIO
    else:
        logger.warning(f"虽然local_file_path有值{local_file_path},不是md、pdf、html或音频类型,所以无法识别,直接跳转到END节点!")
        state["file_type"] = FileType.UNSUPPORTED
        return state

    state["file_title"] = Path(local_file_path).stem
    return state
