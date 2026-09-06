"""
HTML → Markdown 转换模块，负责把 HTML 文本渲染为适合知识库切片的 Markdown。

本模块是纯转换能力：输入 str、输出 str，不感知 LangGraph、不做文件读写、不接触 state。
业务编排（路径校验、编码探测、落盘、更新 state）由 rag 层的 html_parse_service 负责。

转换基于标准库 HTMLParser 自行实现，不额外引入三方依赖。
"""
import re
from html.parser import HTMLParser

from app.shared.runtime.logger import step_log


class HtmlToMarkdownConverter(HTMLParser):
    """
    极简 HTML → Markdown 渲染器。

    只覆盖书籍资料这类常规 HTML：标题、段落、列表、表格、链接、图片、代码块、引用。
    脚本、样式等无关内容直接跳过，避免把 JS/CSS 混进知识库切片。
    """

    _HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    # 只放"内部有文本且必须丢弃"的标签。meta/link 这类空元素没有文本内容，
    # 放进这里反而会因为缺少结束标签让 _skip_depth 只增不减，把整篇正文吞掉。
    _SKIP_TAGS = {"script", "style", "head", "title", "noscript", "iframe"}
    # 需要在边界补换行的块级标签，否则转换后正文会粘连成一坨。
    _BLOCK_TAGS = {
        "p", "div", "section", "article", "header", "footer", "main", "aside", "nav",
        "figure", "figcaption", "blockquote", "table", "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._pre_buffer: list[str] = []
        self._list_stack: list[dict] = []
        self._link_stack: list[tuple[str, list[str]]] = []
        self._table_rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ---------- 输出路由 ----------
    def _emit(self, text: str) -> None:
        """按当前所处上下文把文本写到对应缓冲区。"""
        if not text or self._skip_depth:
            return
        if self._pre_depth:
            self._pre_buffer.append(text)
        elif self._cell is not None:
            self._cell.append(text)
        elif self._link_stack:
            self._link_stack[-1][1].append(text)
        else:
            self._parts.append(text)

    def _newline(self) -> None:
        """块级元素边界补一个换行。"""
        self._parts.append("\n")

    # ---------- 标签回调 ----------
    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._pre_depth:
            # 代码块的缩进和换行属于内容本身，必须原样保留。
            self._emit(data)
            return
        text = re.sub(r"\s+", " ", data)
        if not text.strip():
            return
        self._emit(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        attr_dict = {key.lower(): (value or "") for key, value in attrs}

        if tag == "body":
            # body 出现即代表头部结束，清空跳过计数，
            # 防止 head 里未闭合的标签让后续正文被整段吞掉。
            self._skip_depth = 0
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "br":
            self._emit("\n")
            return
        if tag == "hr":
            self._newline()
            self._emit("\n---\n")
            self._newline()
            return
        if tag == "img":
            src = attr_dict.get("src", "")
            if src:
                self._emit("![{}]({})".format(attr_dict.get("alt", ""), src))
            return
        if tag == "a":
            self._link_stack.append((attr_dict.get("href", ""), []))
            return
        if tag == "pre":
            self._pre_depth += 1
            self._newline()
            return
        # pre 内的 code 不再加行内反引号，整块由 ``` 包裹。
        if tag == "code" and not self._pre_depth:
            self._emit("`")
            return
        if tag in ("strong", "b"):
            self._emit("**")
            return
        if tag in ("em", "i"):
            self._emit("*")
            return
        if tag in self._HEADINGS:
            self._newline()
            self._emit("#" * self._HEADINGS[tag] + " ")
            return
        if tag == "li":
            self._newline()
            indent = "  " * max(len(self._list_stack) - 1, 0)
            if self._list_stack and self._list_stack[-1]["ordered"]:
                self._list_stack[-1]["index"] += 1
                marker = "{}. ".format(self._list_stack[-1]["index"])
            else:
                marker = "- "
            self._emit(indent + marker)
            return
        if tag in ("ul", "ol"):
            self._newline()
            self._list_stack.append({"ordered": tag == "ol", "index": 0})
            return
        if tag == "table":
            self._newline()
            self._table_rows = []
            return
        if tag == "tr":
            self._row = []
            return
        if tag in ("td", "th"):
            self._cell = []
            return
        if tag == "blockquote":
            self._newline()
            self._emit("> ")
            return
        if tag in self._BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "a":
            if not self._link_stack:
                return
            href, text_parts = self._link_stack.pop()
            text = "".join(text_parts).strip()
            # pop 之后再 emit，链接文本才能正确落到单元格或正文里。
            self._emit("[{}]({})".format(text, href) if href else text)
            return
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
            if self._pre_depth == 0:
                code = "".join(self._pre_buffer).strip("\n")
                self._pre_buffer = []
                self._emit("\n```\n" + code + "\n```\n")
            return
        if tag == "code" and not self._pre_depth:
            self._emit("`")
            return
        if tag in ("strong", "b"):
            self._emit("**")
            return
        if tag in ("em", "i"):
            self._emit("*")
            return
        if tag == "table":
            self._render_table()
            self._newline()
            return
        if tag == "tr":
            if self._row is not None:
                self._table_rows.append(self._row)
                self._row = None
            return
        if tag in ("td", "th"):
            if self._cell is not None:
                text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
                if self._row is None:
                    self._row = []
                self._row.append(text)
                self._cell = None
            return
        if tag in ("ul", "ol"):
            if self._list_stack:
                self._list_stack.pop()
            self._newline()
            return
        if tag in self._HEADINGS or tag in self._BLOCK_TAGS:
            self._newline()

    # ---------- 结果输出 ----------
    def _render_table(self) -> None:
        """把收集到的表格行渲染成 Markdown 管道表格。"""
        rows = [row for row in self._table_rows if row]
        if not rows:
            return
        width = max(len(row) for row in rows)
        lines: list[str] = []
        for index, row in enumerate(rows):
            cells = list(row) + [""] * (width - len(row))
            lines.append("| " + " | ".join(self._escape_cell(cell) for cell in cells) + " |")
            if index == 0:
                lines.append("| " + " | ".join(["---"] * width) + " |")
        self._parts.append("\n" + "\n".join(lines) + "\n")

    @staticmethod
    def _escape_cell(text: str) -> str:
        """表格单元格里的竖线和换行会破坏表格结构，需要转义。"""
        return text.replace("|", "\\|").replace("\n", " ")

    def result(self) -> str:
        """返回拼接后的原始 Markdown（未做空行清洗）。"""
        return "".join(self._parts)


@step_log("tidy_markdown")
def tidy_markdown(md_text: str) -> str:
    """
    清洗转换结果，保证切分节点拿到干净规整的 Markdown。

    Args:
        md_text: 转换后的原始 Markdown。

    Returns:
        str: 清理行尾空白并压缩多余空行后的文本。
    """
    md_text = re.sub(r"[ \t]+\n", "\n", md_text)
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)
    lines = [line.rstrip() for line in md_text.split("\n")]
    return "\n".join(lines).strip() + "\n"


@step_log("convert_html_to_markdown")
def convert_html_to_markdown(html_text: str) -> str:
    """
    把 HTML 文本渲染成 Markdown。

    Args:
        html_text: HTML 源文本。

    Returns:
        str: 转换并清洗后的 Markdown 文本。
    """
    parser = HtmlToMarkdownConverter()
    parser.feed(html_text)
    parser.close()
    return tidy_markdown(parser.result())
