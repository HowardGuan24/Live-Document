"""
数据模型 — LearningSpec 及相关结构定义
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
class CausalStep:
    """因果步骤"""
    cause: str
    change: str
    visual_evidence: str


@dataclass
class LearningSpec:
    """
    统一结构化 JSON — 文段的学习解释计划

    字段说明：
    - learning_goal: 本段核心学习主旨
    - entities: 段落中的关键实体（事物、对象）
    - state_variables: 动态过程中的可变参数
    - causal_steps: 因果链（原因 → 变化 → 视觉证据）
    - invariants: 过程中必须满足的约束
    - comprehension_questions: 配套理解检测问题
    """
    learning_goal: Optional[str]
    entities: List[str]
    state_variables: List[str]
    causal_steps: List[dict]  # 每项: {"cause", "change", "visual_evidence"}
    invariants: List[str]
    comprehension_questions: List[str]
    fallback_reason: Optional[str] = None

    def to_json(self, indent=2) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
