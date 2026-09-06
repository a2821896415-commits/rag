from enum import Enum
from typing import TypedDict
import copy


class FileType(str, Enum):
    """导入文件类型，作为导入图路由的单一决策依据（取代原先多个 is_xxx_read_enabled 布尔）。"""
    MD = "md"
    PDF = "pdf"
    HTML = "html"
    AUDIO = "audio"
    UNSUPPORTED = "unsupported"


class ImportGraphState(TypedDict): #为了轻量化和IDE自动补全
    """
    图的定义状态，包含所有节点产生和消费的数据字段
    TypedDict 让我们在代码中能有自动补全和类型检查
    使用字典式访问
    """
    task_id:str

    # 流程控制标记：以单一 file_type 取代原先多个 is_xxx_read_enabled 布尔，
    # 避免用 N 个布尔模拟单选枚举，并让「识别」与「路由」共用同一决策依据。
    file_type: FileType
    id_doc_read_enable:bool
    #加入扩展的文件名

    #路径设置
    local_dir:str
    local_file_path:str
    file_title:str
    pdf_path:str
    md_path:str
    html_path:str
    doc_path:str
    #扩展文件地址

    #内容数据
    md_content:str
    chunks:list
    item_name:str

    #数据库相关
    embeddings_content:list

graph_default_state:ImportGraphState = {
    "task_id":"",
    "file_type":FileType.UNSUPPORTED,
    "local_dir":"",
    "local_file_path":"",
    "file_title":"",
    "pdf_path":"",
    "md_path":"",
    "html_path":"",
    "md_content":"",
    "chunks":[],
    "item_name":"",
    "embeddings_content":[]
}

#暴露了依赖注入（DI）入口。
def create_default_state(**overrides) -> ImportGraphState: #提供两种创建方式，前者可覆盖参数，后者纯粹返回默认。
    """创建默认状态，支持覆盖"""
    #在内存中完全克隆一份全新的、递归隔离的状态字典，让新状态的修改完全不影响旧状态。
    #核心价值：递归隔离
    state = copy.deepcopy(graph_default_state) #设定默认空值，方便创建新实例
    state.update(overrides)
    return state

def get_default_state() -> ImportGraphState:
    """返回一个新的状态实例，避免污染全局变量"""
    return copy.deepcopy(graph_default_state)