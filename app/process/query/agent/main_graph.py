"""
应用主包 / 查询流程兼容层 / 图编排子模块的图编排入口文件，负责组织节点执行顺序与分支流转。
"""
from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

#1. 定义状态图对象,并且指定全局的state
query_graph = StateGraph(QueryGraphState)
#2. 添加节点信息
query_graph.add_node("node_item_name_confirm",node_item_name_confirm)
query_graph.add_node("node_search_embedding",node_search_embedding)
query_graph.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
query_graph.add_node("node_web_search_mcp",node_web_search_mcp)
query_graph.add_node("node_rrf",node_rrf)
query_graph.add_node("node_rerank",node_rerank)
query_graph.add_node("node_answer_output",node_answer_output)

#3. 指定入口节点 (意图守卫已合并进 node_item_name_confirm 的同一 LLM 调用)
query_graph.set_entry_point("node_item_name_confirm")
#4. 指定条件边,动态边
# state answer 进行判定!
# None -> 第一个节点已经顺利的识别出了 item_names 提问没有问题
# str  -> 提问是空 | 有不确定的item_names  | 没有识别对应的item_name
def  node_item_name_confirm_after_router(state:QueryGraphState):
    # 如果前置节点已经生成了澄清或兀底答案，就直接跳转到输出节点收口。
    if state['answer']:
       # 不为空!
       # str  -> 提问是空 | 有不确定的item_names  | 没有识别对应的item_name
       logger.warning(f"node_item_name_confirm_无法继续向后执行: {state['answer']}")
       return "node_answer_output"
    # 为空,可以正常执行,并发执行多路检索节点
    return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"
query_graph.add_conditional_edges("node_item_name_confirm"
                                  ,node_item_name_confirm_after_router,
                                  {
                                      "node_answer_output":"node_answer_output",
                                      "node_search_embedding":"node_search_embedding",
                                      "node_search_embedding_hyde":"node_search_embedding_hyde",
                                      "node_web_search_mcp":"node_web_search_mcp"
                                  })
#5. 指定静态边
query_graph.add_edge("node_search_embedding","node_rrf")
query_graph.add_edge("node_search_embedding_hyde","node_rrf")
query_graph.add_edge("node_web_search_mcp","node_rrf")
query_graph.add_edge("node_rrf","node_rerank")
query_graph.add_edge("node_rerank","node_answer_output")
query_graph.add_edge("node_answer_output",END)
#6. 编译对象即可
query_app = query_graph.compile()
