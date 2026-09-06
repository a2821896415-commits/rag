"""
MOSS 转写服务门面模块，统一封装音频转写相关操作。

对外暴露领域语义接口（transcribe / is_healthy），内部委托 moss_utils，
配置从 infra_config.moss 获取，不直接接触底层 HTTP 与 ffmpeg 细节。
"""
from pathlib import Path

from app.infra.config import infra_config
from app.shared.clients.moss_utils import is_healthy, transcribe_audio


class MossGateway:
    @property
    def enabled(self) -> bool:
        """MOSS 转写服务是否启用。"""
        return infra_config.moss.enabled

    @property
    def base_url(self) -> str:
        """MOSS 转写服务地址。"""
        return infra_config.moss.base_url

    def is_healthy(self) -> bool:
        """探测 MOSS 服务是否就绪。"""
        return is_healthy()

    def transcribe(self, audio_path: str | Path) -> list[dict]:
        """
        转写音频文件，返回合并后的 segments。

        Args:
            audio_path: 音频文件路径。
        Returns:
            list[dict]: MOSS 返回的 segments 列表（大文件已分块拼接）。
        """
        if not self.enabled:
            raise RuntimeError(
                "MOSS 转写服务未启用：请在配置/环境变量中设置 MOSS_SERVICE_ENABLED=true"
            )
        return transcribe_audio(Path(audio_path))


moss_gateway = MossGateway()
