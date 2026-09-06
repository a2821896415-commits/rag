"""
应用主包 / 查询流程兼容层 / 图编排子模块 / 节点适配层中的 node_web_search_mcp 模块，负责承载对应场景的具体实现逻辑。
"""
import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_web_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索引擎补充联网检索结果。
    """
    # 联网检索节点开始后，前端可以看到“网络搜索”已进入执行中。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    # 这里直接调用 rag 层的联网搜索服务，把网页结果补充进后续重排候选集。
    pages = search_web_documents(state, count=10)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {"web_search_docs": pages}


if __name__ == '__main__':
    test_state = {
        "session_id": "xxxx",
        "is_stream": False,
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    }

    result_state = node_web_search_mcp(test_state)
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")
    print(f"查询内容: {result_state}")
