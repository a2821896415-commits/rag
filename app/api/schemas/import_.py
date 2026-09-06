"""
应用主包 / 接口层 / 数据模型层中的 import_ 模块，负责承载对应场景的具体实现逻辑。
"""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    code: int = 200
    message: str
    task_ids: list[str]


class ImportStatusResponse(BaseModel):
    code: int = 200
    task_id: str
    status: str | None = None
    done_list: list[str]
    running_list: list[str]
