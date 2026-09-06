"""
应用主包 / 导入流程兼容层 / 图编排子模块 / 节点适配层中的 node_pdf_to_md 模块，负责承载对应场景的具体实现逻辑。
"""
import os

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.pdf_parse_service import parse_pdf_to_markdown
from app.shared.utils.path_util import PROJECT_ROOT
"""
节点作用: node_pdf_to_md  将pdf转成md,并且保存和存储,同时修改state相关的参数
入参:  [pdf_path:str :Path   local_dir:str :Path 默认的存储文件地址(项目/output) ]
出参:  [md_path:str  md_content:str]
步骤:
   1. 日志+进行中的任务记录 add_running_task
   2. step_1_validate_paths 校验pdf和输出地址
   3. step_2_upload_and_poll minerU进行交互
   4. step_3_download_and_extract 下载提取和解压
   5. 根据md地址读取对应md_content内容,并且更新state
   6. 日志+完成的任务记录  add_done_task
"""
@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    """
    # PDF 解析通常最耗时，因此单独记录开始状态，便于前端展示进度。
    add_running_task(state["task_id"], "node_pdf_to_md")
    # 具体的上传、轮询、下载和落盘都在 rag 层 service 中处理。
    state = parse_pdf_to_markdown(state)
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.shared.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180使用说明书.pdf")
    pdf_name = os.path.join("source", test_pdf_name)
    test_pdf_path = os.path.join(PROJECT_ROOT, pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")
