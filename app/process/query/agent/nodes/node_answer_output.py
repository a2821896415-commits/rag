"""
应用主包 / 查询流程兼容层 / 图编排子模块 / 节点适配层中的 node_answer_output 模块，负责承载对应场景的具体实现逻辑。
"""
import sys
from app.shared.runtime.logger import logger,node_log
from app.rag.query.answer_output_service import produce_answer
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_answer_output")
def node_answer_output(state):
    """
    节点功能：节点层只保留任务状态与编排衔接，真正的答案生成已经下沉到 `answer_output_service`。
    """
    # 最终输出节点开始执行后，说明检索和重排阶段已经全部完成。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 真正的 Prompt 组装、模型调用和历史落库都放在 service 层统一处理。
    state = produce_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state

# def node_answer_output(state):
#     """
#     节点功能：进行过处理可以是流式输出可以整体输出！
#     """
#     print("---node_answer_output 节点处理开始---")
#     add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
#
#     session_id = state["session_id"]
#     is_stream = state.get("is_stream", True)
#     base_answer = state.get("answer") or f"这是关于「{state.get('original_query', '当前问题')}」的测试回答，正在演示打字机流式输出效果。"
#     final_text = ""
#
#     if is_stream:
#         for ch in base_answer:
#             final_text += ch
#             push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
#             time.sleep(0.03)
#         logger.info(f"流式输出完成，总长度: {len(final_text)}")
#     else:
#         final_text = base_answer
#
#     # 执行完毕之前 存储结果
#     set_task_result(session_id,"answer",final_text)
#     add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
#     print("---node_answer_output 节点处理结束---")
#     return {"answer": final_text}


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "type": "milvus",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://local-server/images/panel_view.jpg)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "type": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "type": "milvus",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"},
        {"role": "user", "text": "HAK 180 烫金机","rewritten_query":"HAK 180 烫金机的具体操作步骤和面板设置方法"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
