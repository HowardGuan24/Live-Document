"""
数据模型 — ExplanationSpec 及相关结构定义
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional
import json


class ExplanationType(str, Enum):
    """解释类型"""
    FORMULA = "formula"
    PROCESS = "process"
    DATAFLOW = "dataflow"
    OPERATION = "operation"
    SCENE = "scene"


class Renderer(str, Enum):
    """渲染器类型"""
    MANIM = "manim"
    CSS_ANIMATION = "css_animation"
    THREE_JS = "three_js"
    SVG = "svg"
    TEXT_ONLY = "text_only"


@dataclass
class Segment:
    """文档分段"""
    text: str
    section: str = ""
    index: int = 0


@dataclass
class ExplanationSpec:
    """
    统一结构化 JSON — 提交给下游渲染器的动画计划
    """
    source_text: str
    type: str
    renderer: str
    goal: str
    objects: List[str]
    steps: List[str]
    confidence: float = 1.0
    fallback_reason: Optional[str] = None

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
