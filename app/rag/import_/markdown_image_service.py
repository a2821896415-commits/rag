"""
Markdown 图片增强服务模块，负责图片总结、上传与引用替换。
"""
import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Dict

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from minio.deleteobjects import DeleteObject

from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.infra.object_storage import minio_gateway
from app.shared.utils.rate_limit_utils import apply_api_rate_limit


@step_log("is_supported_image")
def is_supported_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


@step_log("load_markdown_and_image_dir")
def load_markdown_and_image_dir(state: dict) -> tuple[str, Path, Path]:
    md_content = state.get("md_content", "")
    md_path = state.get("md_path")
    if not md_path:
        logger.error("md_path核心参数为空,无法继续!!")
        raise ValueError("md_path核心参数为空,无法继续!!")
    md_path_obj = Path(md_path)
    if not md_content:
        logger.warning("md_content为空,根据地址读取!")
        md_content = md_path_obj.read_text(encoding="utf-8")
        state["md_content"] = md_content
    images_path_obj = md_path_obj.parent / "images"
    return md_content, md_path_obj, images_path_obj


@step_log("scan_images")
def scan_images(md_content: str, images_path_obj: Path) -> list[tuple[str, str, tuple[str, str]]]:
    image_context_list: list[tuple[str, str, tuple[str, str]]] = []
    for image_file in images_path_obj.iterdir():
        image_name = image_file.name
        if not is_supported_image(image_name):
            logger.warning(f"{image_name}不是图片,无需处理,跳过本次!!")
            continue
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        match_obj = rep.search(md_content)
        if not match_obj:
            logger.warning(f"{image_name}没有在md中使用,跳过本次处理!")
            continue
        start, end = match_obj.span()
        pre_context = md_content[max(start - 100, 0):start]
        pos_context = md_content[end:min(end + 100, len(md_content))]
        image_context_list.append((image_name, str(image_file), (pre_context, pos_context)))
    return image_context_list


@step_log("summarize_images")
def summarize_images(image_context_list: list[tuple[str, str, tuple[str, str]]], stem: str) -> Dict[str, str]:
    image_summaries_dict: Dict[str, str] = {}
    vm_model = llm_provider.vision_chat()
    for image_name, image_path_str, context in image_context_list:
        apply_api_rate_limit()
        image_context_prompt = load_prompt("image_summary", root_folder=stem, image_content=context)
        image_path_obj = Path(image_path_str)
        image_data = base64.b64encode(image_path_obj.read_bytes()).decode(encoding="utf-8")
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mimetypes.guess_type(image_name)[0]};base64,{image_data}"
                    },
                },
                {"type": "text", "text": image_context_prompt},
            ]
        )
        summary = (vm_model | StrOutputParser()).invoke([message])
        image_summaries_dict[image_name] = summary
    return image_summaries_dict


@step_log("upload_images_and_replace")
def upload_images_and_replace(
    image_context_list: list[tuple[str, str, tuple[str, str]]],
    image_summaries_dict: Dict[str, str],
    md_content: str,
    stem: str,
) -> str:
    minio_client = minio_gateway.client()
    object_list = minio_client.list_objects(
        bucket_name=minio_gateway.bucket_name,
        prefix=f"{minio_gateway.image_dir[1:]}/{stem}/",
        recursive=True,
    )
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    errors = minio_client.remove_objects(
        bucket_name=minio_gateway.bucket_name,
        delete_object_list=delete_object_list,
    )
    for error in errors:
        logger.warning(f"删除失败,失败原因:{error}")

    image_url_dict: dict[str, str] = {}
    for image_name, image_path_str, _ in image_context_list:
        try:
            minio_client.fput_object(
                bucket_name=minio_gateway.bucket_name,
                object_name=f"{minio_gateway.image_dir}/{stem}/{image_name}",
                file_path=image_path_str,
                content_type=mimetypes.guess_type(image_name)[0],
            )
            image_minio_url = minio_gateway.build_image_url(stem=stem, image_name=image_name)
            image_url_dict[image_name] = image_minio_url
        except Exception:
            logger.warning(f"本次图片上传失败:{image_name},跳过继续上传下一张!")
            continue

    if not image_url_dict:
        logger.warning("图片上传全部失败!")
        return md_content

    for image_name, image_url in image_url_dict.items():
        image_summary = image_summaries_dict[image_name]
        rep = re.compile(r"\!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        md_content = rep.sub(lambda _: f"![{image_summary}]({image_url})", md_content)
    return md_content


@step_log("backup_markdown")
def backup_markdown(new_md_content: str, md_path_obj: Path) -> str:
    new_md_path_obj = md_path_obj.with_name(f"{md_path_obj.stem}_new.md")
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")
    return str(new_md_path_obj)


@step_log("enrich_markdown_images")
def enrich_markdown_images(state: dict) -> dict:
    # 先拿到 Markdown 正文、Markdown 文件路径和图片目录。
    md_content, md_path_obj, images_path_obj = load_markdown_and_image_dir(state)
    if not images_path_obj.exists() or len(list(images_path_obj.iterdir())) == 0:
        logger.warning("图片文件夹为空或者没有图片,无需后续处理!")
        return state
    # 扫描 Markdown 中真正被引用到的图片，并顺带截取上下文。
    image_context_list = scan_images(md_content, images_path_obj)
    logger.info(f"已经获取上下文信息:{image_context_list}")
    # ??????????????????????? alt ???
    image_summaries_dict = summarize_images(image_context_list, md_path_obj.stem)
    # 上传图片到对象存储，并把 Markdown 中的本地引用替换成可访问 URL。
    new_md_content = upload_images_and_replace(image_context_list, image_summaries_dict, md_content, md_path_obj.stem)
    state["md_content"] = new_md_content
    state["md_path"] = backup_markdown(new_md_content, md_path_obj)
    return state
