"""
文档解析器 — 分段与上下文提取
"""

import re
from typing import List
from .models import Segment


def parse_document(text: str) -> List[Segment]:
    """
    将文档拆分为段落，提取所在章节作为上下文。

    规则：
    1. 以 Markdown 标题 (# / ## / ### ...) 标记章节
    2. 连续非空行组成一个段落；空行分隔段落
    3. 每段保留所属章节名
    """
    lines = text.split("\n")
    current_section = ""
    paragraphs: List[List[str]] = []
    current_para: List[str] = []

    for line in lines:
        stripped = line.strip()

        # 检测 Markdown 标题
        heading_match = re.match(r"^(#{1,6})\s+(.+)", stripped)
        if heading_match:
            # 先结束当前段落
            if current_para:
                paragraphs.append((current_section, current_para[:]))
                current_para = []
            current_section = heading_match.group(2).strip()
            continue

        # 空行 = 段落分隔
        if not stripped:
            if current_para:
                paragraphs.append((current_section, current_para[:]))
                current_para = []
            continue

        current_para.append(stripped)

    # 处理最后一个段落
    if current_para:
        paragraphs.append((current_section, current_para[:]))

    # 构造 Segment 列表
    segments = []
    for idx, (section, lines) in enumerate(paragraphs):
        text = " ".join(lines)
        if len(text) < 2:  # 过短的内容不构成有效段落
            continue
        segments.append(Segment(text=text, section=section, index=idx))

    return segments
