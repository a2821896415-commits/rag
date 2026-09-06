"""
MOSS 转写服务配置模块，负责读取音频转写相关环境变量。

MOSS 服务运行在独立 venv（依赖 transformers 5.x），与主项目彻底解耦，
本模块只承载其连接与调用参数，供 infra 层门面与 client 层 utils 使用。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_float, env_str


@dataclass
class MossConfig:
    enabled: bool
    base_url: str
    timeout: float
    max_audio_bytes: int
    max_audio_segment_seconds: int
    model: str


moss_config = MossConfig(
    enabled=env_bool("MOSS_SERVICE_ENABLED", False),
    base_url=env_str("MOSS_SERVICE_BASE_URL", "http://127.0.0.1:8088"),
    timeout=env_float("MOSS_SERVICE_TIMEOUT", 600.0),
    max_audio_bytes=int(env_str("MAX_AUDIO_BYTES", str(5 * 1024 * 1024))),
    max_audio_segment_seconds=int(env_str("MAX_AUDIO_SEGMENT_SECONDS", "600")),
    model=env_str("MOSS_SERVICE_MODEL", "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
)
