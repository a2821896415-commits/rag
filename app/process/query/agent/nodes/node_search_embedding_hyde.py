"""
应用主包 / 查询流程兼容层 / 图编排子模块 / 节点适配层中的 node_search_embedding_hyde 模块，负责承载对应场景的具体实现逻辑。
"""
import sys

from app.shared.runtime.logger import logger, node_log
from app.rag.query.search_embedding_hyde_service import search_embedding_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log(node_name="node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    # 标记 HyDE 检索开始，和普通向量检索形成并行进度展示。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 节点层只负责调度，参数校验和 HyDE 检索流程统一收口到 service。
    mivlus_result = search_embedding_hyde(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": mivlus_result}


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
