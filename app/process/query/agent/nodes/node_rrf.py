"""
应用主包 / 查询流程兼容层 / 图编排子模块 / 节点适配层中的 node_rrf 模块，负责承载对应场景的具体实现逻辑。
"""
import sys
from app.shared.runtime.logger import logger, node_log
from app.rag.query.rrf_service import fuse_retrieval_results
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rrf")
def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    # RRF 开始时登记进度，说明多路结果即将进入融合排序。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 把普通检索和 HyDE 检索统一融合，生成后续重排的候选集。
    state["rrf_chunks"] = fuse_retrieval_results(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"]
    }

    try:
        from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
        from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
