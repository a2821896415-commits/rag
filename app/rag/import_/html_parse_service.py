"""
HTML 解析服务模块，负责把 HTML 文件转换为 Markdown 文本。

与 pdf_parse_service 的分工一致：本模块只做"解析成 Markdown 并落盘"，
不感知 LangGraph，节点层（node_html_to_md）负责调度与任务状态上报。
HTML → Markdown 的标签渲染细节已抽到 shared 层的 html_to_markdown 模块，
本模块只保留路径校验、编码探测与落盘编排。
"""
import re
from pathlib import Path

from app.shared.runtime.logger import logger, step_log
from app.shared.utils.html_to_markdown import convert_html_to_markdown
from app.shared.utils.path_util import PROJECT_ROOT

# 编码探测失败时的兜底顺序：gb18030 兼容 gbk/gb2312，覆盖国内站点导出的 HTML。
FALLBACK_ENCODINGS = ("utf-8", "gb18030", "big5", "latin-1")

@step_log("validate_html_paths")
def validate_html_paths(state: dict) -> tuple[Path, Path]:
    """
    校验 HTML 文件路径与输出目录是否可用。

    Args:
        state: 导入图当前状态，需能解析出 HTML 路径。

    Returns:
        tuple[Path, Path]: 依次为 HTML 文件对象与本地输出目录对象。
    """
    html_path = state.get("html_path")
    local_dir = state.get("local_dir")
    if not html_path:
        logger.error("html_path/doc_path/local_file_path均为空,无法读取文件!")
        raise ValueError("html_path/doc_path/local_file_path均为空,无法读取文件!")
    if not local_dir:
        logger.warning("没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    html_path_obj = Path(html_path)
    local_dir_obj = Path(local_dir)
    if not html_path_obj.exists():
        logger.error(f"html_path:{html_path_obj},但是没有文件存在!")
        raise FileNotFoundError(f"html_path:{html_path_obj},但是没有文件存在!")
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir_obj}地址没有文件夹,我们需要主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return html_path_obj, local_dir_obj


@step_log("detect_html_encoding")
def detect_html_encoding(raw: bytes) -> str:
    """
    从 HTML 头部的 charset 声明中探测编码。

    Args:
        raw: HTML 文件原始字节。

    Returns:
        str: 探测到的编码名称，未声明时返回空字符串。
    """
    head = raw[:4096].decode("ascii", errors="ignore")
    matched = re.search(r'charset\s*=\s*["\']?\s*([\w-]+)', head, re.IGNORECASE)
    return matched.group(1) if matched else ""


@step_log("read_html_text")
def read_html_text(html_path_obj: Path) -> str:
    """
    读取 HTML 源文本，自动处理常见中文编码。

    Args:
        html_path_obj: HTML 文件对象。

    Returns:
        str: 解码后的 HTML 文本。
    """
    raw = html_path_obj.read_bytes()
    declared = detect_html_encoding(raw)
    if declared:
        try:
            return raw.decode(declared)
        except (UnicodeDecodeError, LookupError):
            logger.warning(f"HTML声明编码:{declared}解码失败,改用兜底编码列表!")
    for encoding in FALLBACK_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.warning("所有编码均解码失败,采用utf-8忽略非法字符!")
    return raw.decode("utf-8", errors="ignore")


@step_log("parse_html_to_markdown")
def parse_html_to_markdown(state: dict) -> dict:
    """
    HTML 转 Markdown 主入口，与 parse_pdf_to_markdown 保持同样的状态契约。

    Args:
        state: 导入图当前状态，需能解析出 HTML 路径与输出目录。

    Returns:
        dict: 补充 `md_path` 与 `md_content` 后的最新状态。
    """
    # 先校验 HTML 路径和输出目录，避免把非法输入送进解析流程。
    html_path_obj, local_dir_path_obj = validate_html_paths(state)
    logger.info(f"开始解析HTML文件:{html_path_obj}")

    html_text = read_html_text(html_path_obj)
    md_text = convert_html_to_markdown(html_text)

    md_path_obj = local_dir_path_obj / "{}.md".format(html_path_obj.stem)
    md_path_obj.write_text(md_text, encoding="utf-8")

    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_text
    logger.info(f"HTML转Markdown完成,输出:{md_path_obj},字符数:{len(md_text)}")
    return state
