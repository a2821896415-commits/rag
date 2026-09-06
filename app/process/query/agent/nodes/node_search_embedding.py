"""
应用主包 / 查询流程兼容层 / 图编排子模块 / 节点适配层中的 node_search_embedding 模块，负责承载对应场景的具体实现逻辑。
"""
import sys

from app.shared.runtime.logger import logger, node_log
from app.rag.query.search_embedding_service import search_embedding
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log(node_name="node_search_embedding")
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    # 先记录节点开始，和 SSE 进度展示保持一致。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 这里只保留节点调度职责，参数校验和检索执行统一交给 service 入口。
    mivlus_result = search_embedding(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"embedding_chunks": mivlus_result}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机# 对于本设备所有者不遵守本指南中规定的说明操作而导致的损害，Brother 不承担任何责任。",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果")
        print(f"\n>>> 测试完成！检索到 {chunks} 条结果")
    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)
