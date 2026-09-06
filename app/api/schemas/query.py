"""
应用主包 / 接口层 / 数据模型层中的 query 模块，负责承载对应场景的具体实现逻辑。
"""
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询内容")
    session_id: str | None = Field(default=None, description="会话ID，为空时自动生成")
    is_stream: bool = Field(default=False, description="是否使用SSE流式输出")
    role: str = Field(default="", description="用户角色：听众/内容运营/内容编辑")


class SourceItem(BaseModel):
    title: str = ""
    book_name: str = ""
    author: str = ""
    content_type: str = ""
    file_title: str = ""
    source: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    message: str
    session_id: str
    answer: str = ""
    done_list: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)


class HistoryItem(BaseModel):
    id: str = Field(default="", alias="_id")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    ts: Any = None


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItem] = Field(default_factory=list)
