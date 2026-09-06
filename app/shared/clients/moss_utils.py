"""
MOSS 转写服务 client 工具模块，负责底层 HTTP 调用与音频切片细节。

MOSS 服务运行在独立 venv（依赖 transformers 5.x），与本主项目彻底解耦，
本模块只通过 HTTP 与之交互：维护会话单例、按需用 ffmpeg 切片大音频、调用 /transcribe。
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests

from app.shared.config.moss_config import moss_config
from app.shared.runtime.logger import logger

# 全局 requests.Session 单例，复用 TCP 连接。
_moss_session: requests.Session | None = None


def get_moss_session() -> requests.Session:
    """MOSS 服务 HTTP 会话单例，复用连接。"""
    global _moss_session
    if _moss_session is None:
        _moss_session = requests.Session()
    return _moss_session


def _ffmpeg_bin(name: str) -> str | None:
    """
    返回 ffmpeg / ffprobe 可执行文件路径，按以下顺序查找：
    1) imageio-ffmpeg 随包提供的静态二进制（位于主项目 venv 内，与 MOSS 独立 venv 解耦）
    2) 项目内置版本 <app/shared>/tools/ffmpeg/bin/<name>.exe（基于模块自身位置解析，随包移动）
    3) PATH 中可发现的版本

    主项目 venv 内自带的二进制优先，确保不与 MOSS 的 .venv-moss 互相影响。
    """
    if name not in ("ffmpeg", "ffprobe"):
        return shutil.which(name)
    try:
        import imageio_ffmpeg
        exe = (
            imageio_ffmpeg.get_ffmpeg_exe()
            if name == "ffmpeg"
            else imageio_ffmpeg.get_ffprobe_exe()
        )
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    bundled = (
        Path(__file__).resolve().parents[1] / "tools" / "ffmpeg" / "bin" / f"{name}.exe"
    )
    if bundled.exists():
        return str(bundled)
    return shutil.which(name)


def is_healthy() -> bool:
    """探测 MOSS 服务是否就绪（GET /health）。"""
    try:
        resp = get_moss_session().get(f"{moss_config.base_url}/health", timeout=5)
        return resp.status_code == 200 and resp.json().get("status") == "ok"
    except Exception:
        return False


def _transcribe_one(
    audio_path: Path, max_new_tokens: int | None = None
) -> list[dict]:
    """调用 MOSS /transcribe 转写单个音频文件，返回 segments 列表。

    Args:
        audio_path: 音频文件路径。
        max_new_tokens: 可选，覆盖服务端默认的生成 token 上限（用于长音频防截断）。
    """
    payload: dict[str, object] = {"audio_path": str(audio_path)}
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    try:
        resp = get_moss_session().post(
            f"{moss_config.base_url}/transcribe",
            data=payload,
            timeout=moss_config.timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"调用MOSS转写服务失败:{exc}")
        raise RuntimeError(
            f"调用MOSS转写服务失败({moss_config.base_url}),请确认独立服务已启动。错误:{exc}"
        ) from exc
    return resp.json().get("segments", [])


def _probe_duration_av(audio_path: Path) -> float:
    """用 PyAV 直接读取容器时长（秒），失败返回 0.0。

    作为 ffmpeg 文本解析的兜底：不依赖子进程，也不受控制台编码影响。
    """
    try:
        import av

        with av.open(str(audio_path)) as container:
            if container.duration is None:
                return 0.0
            return float(container.duration) / 1_000_000.0
    except Exception:
        return 0.0


def _probe_duration(audio_path: Path) -> float:
    """用 ffmpeg 解析音频时长（秒）。ffprobe 可能不存在，故直接用 ffmpeg。

    不可用或失败返回 0.0。
    """
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if not ffmpeg:
        return 0.0
    try:
        # 不能用 text=True：Windows 下它按 GBK 解码 ffmpeg 输出，
        # 遇到非 GBK 字节会在读取线程抛 UnicodeDecodeError，导致 stderr 变成 None。
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(audio_path)],
            capture_output=True, timeout=30,
        )
    except (subprocess.SubprocessError, OSError):
        return 0.0
    # ffmpeg 把媒体信息打印到 stderr，形如 "Duration: 00:05:23.45, start: ..."
    raw = out.stderr or b""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        return _probe_duration_av(audio_path)
    h, mi, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def _estimate_max_new_tokens(duration: float) -> int:
    """按音频时长估算生成上限，避免长音频被 max_new_tokens 静默截断。

    实测约每分钟音频产生 300~450 个输出 token，这里按 500/分钟并留 512 余量。
    """
    if duration <= 0:
        return 2048
    return int(min(16384, max(2048, duration / 60.0 * 500 + 512)))


def _split_audio_by_size(
    audio_path: Path, max_bytes: int, duration: float = 0.0
) -> list[tuple[Path, float]]:
    """
    用 ffmpeg 将音频切成多段临时文件，段长取【双约束的更保守者】。

    两个约束：
    1) 时长上限 max_audio_segment_seconds：MOSS 按音频序列长度消耗显存，
       单段过长会 oom。
    2) 字节上限 max_bytes：按平均比特率折算为时长，避免单段文件过大。

    只取其一会脱节：例如"字节超限触发分块，但总时长不足时长上限"时，
    若只按时长切会只切出 1 段（等于没切）。

    返回 [(片段路径, 该片段在时间轴上的起始偏移秒), ...]。
    若无法切片（无 ffmpeg 或无法探测时长），返回 [(原文件, 0.0)]，由调用方整段转写。
    """
    if duration <= 0:
        duration = _probe_duration(audio_path)
    ffmpeg = _ffmpeg_bin("ffmpeg")
    if not ffmpeg or duration <= 0:
        logger.warning("无法探测音频时长或未找到 ffmpeg，回退为整段转写")
        return [(audio_path, 0.0)]

    seg_seconds = float(moss_config.max_audio_segment_seconds)
    size = audio_path.stat().st_size
    # 按平均比特率把字节上限折算为时长，留 10% 余量。
    byte_seg = duration * max_bytes / size * 0.9 if size > 0 else seg_seconds
    target = max(1.0, min(seg_seconds, byte_seg))
    logger.info(
        f"切片策略: 时长上限 {seg_seconds:.0f}s / 字节估算 {byte_seg:.0f}s "
        f"→ 采用 {target:.0f}s"
    )
    suffix = audio_path.suffix or ".wav"
    chunks: list[tuple[Path, float]] = []
    t = 0.0
    idx = 0
    while t < duration:
        seg_len = min(target, duration - t)
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        logger.info(
            f"ffmpeg 切片 [{idx + 1}]: {t:.1f}s ~ {t + seg_len:.1f}s (时长 {seg_len:.1f}s)"
        )
        try:
            subprocess.run(
                [ffmpeg, "-y", "-ss", f"{t:.3f}", "-i", str(audio_path),
                 "-t", f"{seg_len:.3f}", "-vn", "-acodec", "copy", tmp],
                capture_output=True, timeout=120,
            )
        except subprocess.SubprocessError as exc:
            logger.warning(f"ffmpeg 切片失败@{t:.1f}s: {exc}")
            if os.path.exists(tmp):
                os.unlink(tmp)
            break
        if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            logger.warning(f"ffmpeg 切片产出空文件@{t:.1f}s，停止切片")
            break
        chunks.append((Path(tmp), t))
        t += seg_len
        idx += 1
    logger.info(f"音频切片完成，共 {len(chunks)} 段")
    return chunks or [(audio_path, 0.0)]


def transcribe_audio(audio_path: Path) -> list[dict]:
    """
    转写音频文件，自动按体积上限分块并拼接 segments（时间戳做偏移补偿）。

    Args:
        audio_path: 音频文件路径。
    Returns:
        list[dict]: 合并后的 segments 列表。
    """
    audio_path = Path(audio_path)
    file_size = audio_path.stat().st_size
    seg_seconds = float(moss_config.max_audio_segment_seconds)
    duration = _probe_duration(audio_path)
    # 双维度：字节超限（大文件）或时长超限（长音频）都需分块。
    if file_size <= moss_config.max_audio_bytes and duration <= seg_seconds:
        return _transcribe_one(audio_path)

    logger.info(
        f"音频 {file_size} 字节 / {duration:.1f}s 超过单段上限"
        f"({moss_config.max_audio_bytes} 字节 或 {seg_seconds:.0f}s), 启用分块转写并拼接"
    )
    segments: list[dict] = []
    chunks = _split_audio_by_size(audio_path, moss_config.max_audio_bytes, duration)
    total = len(chunks)
    for i, (chunk_path, offset) in enumerate(chunks, 1):
        size = chunk_path.stat().st_size if chunk_path.exists() else 0
        logger.info(f"开始转写第 {i}/{total} 段: {chunk_path.name} ({size} 字节)")
        # 估算该分片的生成 token 上限，长音频/大分片时增大以防截断。
        chunk_dur = _probe_duration(chunk_path) if chunk_path.exists() else duration
        max_new_tokens = _estimate_max_new_tokens(chunk_dur)
        for seg in _transcribe_one(chunk_path, max_new_tokens):
            seg = dict(seg)
            seg["start"] = (seg.get("start") or 0) + offset
            seg["end"] = (seg.get("end") or 0) + offset
            segments.append(seg)
        if chunk_path != audio_path and chunk_path.exists():
            try:
                chunk_path.unlink()
            except OSError:
                pass
    logger.info(f"分块转写完成，共 {len(segments)} 个 segment")
    return segments
