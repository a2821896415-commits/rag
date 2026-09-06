"""
应用主包 / 导入流程兼容层 / 图编排子模块 / 节点适配层中的 node_document_split 模块，负责承载对应场景的具体实现逻辑。
"""
import os

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import split_document

@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    """
    # 文档切分是后续向量化的前置步骤，因此先把当前进度推到任务列表。
    add_running_task(state["task_id"], "node_document_split")
    # 先按标题粗切，再做长度细切，最终得到适合检索的 chunks。
    state = split_document(state)
    add_done_task(state["task_id"], "node_document_split")
    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.import_.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"source\doc", "活着简介.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "活着简介",
            "local_dir":os.path.join(PROJECT_ROOT, r"source\doc\活着简介"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        # result_state = node_md_img(test_state)
        # logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        # logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
