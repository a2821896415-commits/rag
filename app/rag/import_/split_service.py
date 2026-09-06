"""
文档切块服务模块，负责 Markdown 清洗、标题切分与二次细分。
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.shared.runtime.logger import logger, step_log
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE


@step_log("load_markdown_content")
def load_markdown_content(state: dict) -> tuple[str, str]:
    md_content = state.get("md_content", "")
    file_title = state.get("file_title", "")
    md_path = state.get("md_path", "")

    if not md_content:
        logger.warning("没有从state读取到md_content内容,我们使用md_path尝试再次读取!")
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content
        if not md_content:
            raise ValueError("md_content没数据,并且尝试读取md_path依然没有数据,终止执行!!")

    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return md_content, file_title


@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    reg = re.compile(r"^\s*#{1,6}\s.+")
    lines = md_content.split("\n")
    chunks: list[dict] = []
    current_title = None
    current_title_lines: list[str] = []
    is_code_block = False
    chunk_size = 0

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line)
            continue

        if reg.match(line) and not is_code_block:
            if current_title and len(current_title_lines) > 1:
                chunks.append(
                    {
                        "content": "\n".join(current_title_lines),
                        "title": current_title,
                        "file_title": file_title,
                    }
                )
            current_title = line
            current_title_lines = [current_title]
            chunk_size += 1
        else:
            current_title_lines.append(line)

    if current_title:
        chunks.append(
            {
                "content": "\n".join(current_title_lines),
                "title": current_title,
                "file_title": file_title,
            }
        )
    if chunk_size == 0:
        chunks.append({"content": md_content, "title": "default", "file_title": file_title})
        chunk_size = 1
    logger.info(f"完成语义切割,切块数量:{chunk_size},内容:{chunks[:3]}")
    return chunks


def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    【辅助函数】超长章节二次切分（核心适配LangChain分割器）
    功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
    切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
    :param section: 原始章节字典，必须包含content键，可选title/file_title等
    :param max_length: 单个Chunk最大字符长度，默认使用全局配置
    :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
    """
    # 内容空值兜底：无内容直接返回原章节
    content = section.get("content", "") or ""
    # 长度未超限，无需切分，直接返回原章节（列表格式保持统一）
    if len(content) <= max_length:
        return [section]

    # 标准化预处理：统一换行符，避免不同系统(\r\n/\n)导致的切分异常
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # 提取章节标题，用于组装子Chunk前缀（保留标题上下文）
    title = section.get("title", "") or ""
    # 标题前缀：带空行分隔，与正文区分开
    prefix = f"{title}\n\n" if title else ""
    # 计算正文可用长度：总长度 - 标题前缀长度（避免标题占满Chunk额度）
    available_len = max_length - len(prefix)
    # 极端情况：标题长度超过阈值，无法切分，返回原章节
    if available_len <= 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
        return [section]

    # 清理正文重复标题：避免原章节中正文开头重复标题，导致子Chunk内容冗余
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    # 初始化LangChain递归分割器（核心工具：按优先级分隔符切分，保留语义）
    # separators：分割符优先级（从粗到细），优先按大语义单元切分，最后才硬拆
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_len,  # 正文部分最大长度（已扣除标题）
        chunk_overlap=0,           # 无重叠：按标题切分后语义完整，无需重叠
        # 分割符优先级：空行(段落)→换行→中文标点→英文标点→空格，最后硬拆
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
    )

    # 切分正文并组装子章节（带完整元信息，便于溯源）
    sub_sections: list[dict[str, Any]] = []
    for idx, chunk in enumerate(splitter.split_text(body), start=1):
        # 清理空内容：跳过切分后的空字符串
        text = chunk.strip()
        if not text:
            continue
        # 组装子Chunk完整内容：标题前缀 + 切分后的正文
        full_text = (prefix + text).strip()
        # 子章节元信息：保留父级关联，添加序号，便于后续检索/溯源
        # 先继承父章节已有字段（元数据广播的 book_name/author/category 等），再覆盖切分后的字段。
        sub_sections.append({
            **section,
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",  # 子Chunk标题（带序号）
            "content": full_text,                                     # 切分后的完整内容
            "parent_title": title,                                    # 父章节标题（用于后续合并）
            "part": idx,                                              # 子Chunk序号
            "file_title": section.get("file_title"),                  # 所属文件标题
        })

    logger.debug(f"超长章节切分完成：{title} → 生成{len(sub_sections)}个子Chunk")
    return sub_sections

def _merge_short_sections(
    sections: list[dict[str, Any]],
    min_length: int = CHUNK_SIZE,
    max_length: int = CHUNK_MAX_SIZE,
) -> list[dict[str, Any]]:
    """
    【辅助函数】过短章节合并（减少碎片化，提升检索效果）
    核心规则：仅合并「同父标题」且「当前块长度不足阈值」的相邻Chunk，避免跨章节合并
    :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
    :param min_length: 最小长度阈值，低于此值的Chunk会被合并
    :return: 合并后的Chunk列表，长度适中，保留元信息
    """
    # 边界处理：空列表直接返回，避免后续索引报错
    if not sections:
        logger.debug("待合并Chunk列表为空，直接返回")
        return []

    merged_sections: list[dict[str, Any]] = []  # 最终合并结果
    current_chunk: dict[str, Any] | None = None  # 迭代累加器：保存当前待合并的Chunk

    for sec in sections:
        # 初始化：第一个Chunk直接作为当前待合并块
        if current_chunk is None:
            current_chunk = sec
            continue

        # 合并条件：1.当前块长度不足阈值 2.与下一块同父标题（同属一个原章节）
        current_content = current_chunk.get("content", "")
        is_current_short = len(current_content) < min_length
        is_same_parent = current_chunk.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            # 合并前清理：去掉下一块开头重复的父标题，避免内容冗余
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()
            # 合并时额外校验长度，避免“先切短再合并”后重新超过最大阈值。
            merged_content = current_content + "\n\n" + next_content
            will_exceed_max = max_length > 0 and len(merged_content) > max_length
            if will_exceed_max:
                merged_sections.append(current_chunk)
                current_chunk = sec
                continue

            # 合并内容：空行分隔，保证格式整洁
            current_chunk["content"] = merged_content
            # 更新子Chunk序号：保留最新序号，便于溯源
            if "part" in sec:
                current_chunk["part"] = sec["part"]
            logger.debug(f"合并短Chunk：{current_chunk.get('parent_title')} → 累计长度{len(current_chunk['content'])}")
        else:
            # 不满足合并条件：将当前块加入结果，切换为新的待合并块
            merged_sections.append(current_chunk)
            current_chunk = sec

    # 循环结束后，将最后一个待合并块加入结果
    if current_chunk is not None:
        merged_sections.append(current_chunk)

    logger.debug(f"短Chunk合并完成：原{len(sections)}个 → 合并后{len(merged_sections)}个")
    return merged_sections

@step_log("refine_chunks")
def refine_chunks(
    sections: list[dict],
    max_len: int = CHUNK_MAX_SIZE,
    min_len: int = CHUNK_SIZE,
) -> list[dict]:
    """
        【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
        执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
        :param sections: 步骤3处理后的章节列表
        :param max_len: 单个Chunk最大字符长度
        :return: 长度适中、低碎片化的最终Chunk列表
        """
    # 边界处理：最大长度无效（空/≤0），直接返回原章节，避免切分异常
    if not max_len or max_len <= 0:
        logger.warning(f"步骤4：Chunk最大长度配置无效（{max_len}），跳过精细化处理")
        return sections

    # 阶段1：切分超长章节 → 所有章节长度控制在max_len内
    refined_split = []
    for sec in sections:
        # 对每个章节执行超长切分，结果平铺加入列表（避免嵌套）
        # extend 的作用就是： 把另一个列表（或可迭代对象）里的“元素”，一个个拆出来，直接追加到当前列表的尾部
        refined_split.extend(_split_long_section(sec, max_len))
    logger.info(f"步骤4-1：超长章节切分完成，共生成{len(refined_split)}个初始子Chunk")

    # 阶段2：合并过短章节 → 减少碎片化，提升后续检索/大模型调用效果
    # final_sections = _merge_short_sections(refined_split, min_length=min_len, max_length=max_len)
    # logger.info(f"步骤4-2：过短章节合并完成，最终得到{len(final_sections)}个Chunk")

    # 阶段3：父标题兜底 → 适配Milvus向量库schema（parent_title为必填字段）
    # 兜底规则：无parent_title则用自身title，title也无则填空字符串
    for sec in refined_split:
        if not isinstance(sec, dict):
            continue

        # 补全缺失的part字段（默认0），适配Milvus schema
        if "part" not in sec:
            sec["part"] = 0

        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
    logger.debug(f"步骤4-3：父标题兜底完成，所有Chunk均包含parent_title字段")

    return refined_split


@step_log("backup_chunks")
def backup_chunks(chunks: list[dict], md_path: str) -> None:
    chunks_json_path = Path(md_path).parent / "chunks.json"
    chunks_json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")


def compute_chunk_uid(chunk: dict) -> str:
    """
    为切块生成稳定且确定性的唯一标识，供评估 golden 与 Milvus 精确锚定。

    仅依赖切分阶段即确定的字段（file_title / parent_title / part / content），
    不依赖后续 item_name 赋值，因此重入库后 uid 保持一致、可跨进程复现。
    """
    content = chunk.get("content", "") or ""
    content_hash = hashlib.sha1(content.encode("utf-8")).hexdigest()
    key = "\x00".join([
        str(chunk.get("file_title", "")),
        str(chunk.get("parent_title", "")),
        str(chunk.get("part", 0)),
        str(len(content)),
        content_hash,
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def test(chunks: list[dict]) -> list[dict]:
    # 元数据广播：书名/作者名/条目名称/类别/有声书时长/演播者只写在元数据段里，
    # 解析出来后写回该文件所有 chunk，保证每段都能单独过滤与溯源。
    metadata: dict[str, str] = {}
    for chunk in chunks:
        title = (chunk.get("title") or "").lstrip("#").strip()
        content = chunk.get("content") or ""
        if title != "元数据" and "书名：" not in content:
            continue
        # 元数据形如 "- 书名：三体"，按中文冒号解析成键值对。
        for line in content.split("\n"):
            raw_line = line.strip().lstrip("-").strip()
            if "：" not in raw_line:
                continue
            key, value = raw_line.split("：", 1)
            metadata[key.strip()] = value.strip()
        break

    book_name = metadata.get("书名", "").strip()
    author = metadata.get("作者名", "").strip()
    entry_name = metadata.get("条目名称", "").strip()

    # 类别支持中英文逗号、顿号、分号分隔，拆成标签列表便于按类别过滤。
    category: list[str] = []
    normalized = metadata.get("类别/标签", "").replace("，", ",").replace("、", ",").replace("；", ",").replace(";", ",")
    for tag in normalized.split(","):
        tag = tag.strip()
        if tag and tag not in category:
            category.append(tag)

    broadcast_chunks: list[dict] = []
    for chunk in chunks:
        title = (chunk.get("title") or "").lstrip("#").strip()
        content = chunk.get("content") or ""
        # 元数据段只用于广播，不单独作为知识片段入库。
        if title == "元数据" or "书名：" in content:
            continue
        chunk["book_name"] = book_name
        chunk["author"] = author
        chunk["entry_name"] = entry_name
        chunk["category"] = category
        # 内容类型取章节标题（去井号），二次切分后用父标题兜底，保证取值稳定。
        chunk["content_type"] = (chunk.get("parent_title") or title or "").lstrip("#").strip()
        # 稳定唯一标识，供评估 golden 与 Milvus 精确锚定（见 P0-2）。
        chunk["chunk_uid"] = compute_chunk_uid(chunk)
        broadcast_chunks.append(chunk)

    logger.info(f"元数据广播完成,保留{len(broadcast_chunks)}段,书名:{book_name},作者:{author},条目:{entry_name},类别:{category}")
    return broadcast_chunks


@step_log("split_document")
def split_document(state: dict) -> dict:
    # 先准备好切分所需的 Markdown 正文和文件标题。
    md_content, file_title = load_markdown_content(state)
    # 第一步按标题做语义级粗切，尽量保证一个主题块落在一起。
    chunks = split_by_titles(md_content, file_title)
    # 第二步再按长度细切，避免单块过长影响后续 embedding 和检索。
    chunks = refine_chunks(chunks, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)
    #对chunks进行处理，提取元数据广播
    chunks = test(chunks)
    # 切分结果额外落一份 chunks.json，方便排查切片质量。
    backup_chunks(chunks, state["md_path"])
    state["chunks"] = chunks
    return state
