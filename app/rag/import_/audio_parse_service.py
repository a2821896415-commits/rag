"""
音频解析服务模块，负责调用独立的 MOSS 转写服务（经门面）完成音频到 Markdown 的转换。

MOSS 服务运行在独立 venv（依赖 transformers 5.x），与本主项目（transformers 4.x）彻底解耦，
本模块只做"调用门面 -> 组织成 Markdown -> 落盘"的业务编排，不感知底层 HTTP/ffmpeg 细节。
"""
from pathlib import Path

from app.infra.transcription.moss_gateway import moss_gateway
from app.shared.runtime.logger import logger, step_log
from app.shared.utils.path_util import PROJECT_ROOT


def validate_audio_paths(state):
    audio_path = state.get("local_file_path") or state.get("doc_path") or ""
    local_dir = state.get("local_dir")
    if not audio_path:
        logger.error("local_file_path/doc_path均为空,无法读取音频文件!")
        raise ValueError("local_file_path/doc_path均为空,无法读取音频文件!")
    if not local_dir:
        logger.warning("没有传入local_dir地址,给与默认值!")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    audio_path_obj = Path(audio_path)
    local_dir_obj = Path(local_dir)
    if not audio_path_obj.exists():
        logger.error(f"audio_path:{audio_path_obj},但是没有文件存在!")
        raise FileNotFoundError(f"audio_path:{audio_path_obj},但是没有文件存在!")
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir_obj}地址没有文件夹,我们需要主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)
    return audio_path_obj, local_dir_obj

def _format_timestamp(seconds: float) -> str:
    """把秒数格式化为 [HH:MM:SS] 形式的时间戳。"""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _segments_to_markdown(stem: str, segments: list[dict]) -> str:
    """把 MOSS 返回的 segments 渲染成 Markdown 文本。"""
    lines = [f"# {stem}", "", "## 转录内容", ""]
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = _format_timestamp(seg.get("start", 0))
        end = _format_timestamp(seg.get("end", 0))
        speaker = seg.get("speaker") or ""
        speaker_tag = f" {speaker}" if speaker else ""
        lines.append(f"[{start} - {end}]{speaker_tag}: {text}")
    return "\n".join(lines).strip() + "\n"

@step_log("audio_parse_to_md")
def audio_parse_to_md(state: dict) -> dict:
    """
    音频转 Markdown 主入口。

    通过 MOSS 门面调用独立转写服务，不直接依赖 transformers 5.x，也不感知 HTTP/ffmpeg 细节。

    Args:
        state: 导入图当前状态，需能从 local_file_path / doc_path 解析出音频路径与输出目录。

    Returns:
        dict: 补充 `md_path` 与 `md_content` 后的最新状态。
    """
    audio_path_obj, local_dir_obj = validate_audio_paths(state)
    logger.info(f"调用MOSS服务转写音频文件:{audio_path_obj}")
    segments = moss_gateway.transcribe(audio_path_obj)
    md_text = _segments_to_markdown(audio_path_obj.stem, segments)

    md_path_obj = local_dir_obj / "{}.md".format(audio_path_obj.stem)
    md_path_obj.write_text(md_text, encoding="utf-8")

    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_text
    logger.info(f"音频转Markdown完成,输出:{md_path_obj},字符数:{len(md_text)}")
    return state
