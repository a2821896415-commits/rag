"""
应用主包 / 导入流程兼容层 / 图编排子模块 / 节点适配层中的 node_html_to_md 模块，负责承载对应场景的具体实现逻辑。
"""
import os

from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState, create_default_state
from app.rag.import_.html_parse_service import parse_html_to_markdown
from app.shared.utils.path_util import PROJECT_ROOT
"""
节点作用: node_html_to_md  将html转成md,并且保存和存储,同时修改state相关的参数
入参:  [html_path/doc_path/local_file_path:str :Path   local_dir:str :Path 默认的存储文件地址(项目/output) ]
出参:  [md_path:str  md_content:str]
步骤:
   1. 日志+进行中的任务记录 add_running_task
   2. step_1_validate_paths 校验html和输出地址
   3. step_2_read_and_convert 读取HTML并按编码解码,解析标签渲染成Markdown
   4. step_3_write_md 落盘md文件,并且更新state
   5. 日志+完成的任务记录  add_done_task
"""
@node_log("node_html_to_md")
def node_html_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: HTML转Markdown (node_html_to_md)
    为什么叫这个名字: 核心任务是将 HTML 半结构化数据转换为 Markdown 结构化数据。
    """
    # HTML 解析涉及编码探测与全文扫描，单独记录进度便于前端展示。
    add_running_task(state["task_id"], "node_html_to_md")
    # 具体的路径校验、解码、标签转换与落盘都在 rag 层 service 中处理。
    state = parse_html_to_markdown(state)
    add_done_task(state["task_id"], "node_html_to_md")
    return state

if __name__ == "__main__":

    # 单元测试：验证HTML转MD全流程
    logger.info("===== 开始node_html_to_md节点单元测试 =====")

    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_html_name = os.path.join("doc", "蛤蟆先生去看心理医生简介.html")
    html_name = os.path.join("source", test_html_name)
    test_html_path = os.path.join(PROJECT_ROOT, html_name)

    if not os.path.exists(test_html_path):
        logger.warning(f"测试文件不存在，跳过本次测试：{test_html_path}")
    else:
        # 构造测试状态：doc_path 是状态里给非 md/pdf 文档预留的字段，HTML 走这个入口
        test_state = create_default_state(
            task_id="test_html2md_task_001",
            doc_path=test_html_path,
            local_dir=os.path.join(PROJECT_ROOT, "output")
        )

        result_state = node_html_to_md(test_state)

        logger.info(f"生成Markdown路径：{result_state.get('md_path')}")
        logger.info(f"Markdown内容预览：\n{str(result_state.get('md_content', ''))[:500]}")

    logger.info("===== 结束node_html_to_md节点单元测试 =====")
