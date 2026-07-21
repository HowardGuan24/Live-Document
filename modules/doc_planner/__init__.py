"""
doc_planner — 文档理解与动画规划模块

从文档到 ExplanationSpec 的全流程：解析 → 评分 → 分类 → 路由 → 生成
"""

from .models import ExplanationSpec, ExplanationType, Renderer, Segment
from .parser import parse_document
from .scorer import score_segment, is_worth_dynamicizing
from .classifier import classify_segment
from .router import route_renderer
from .generator import generate_spec

__all__ = [
    "ExplanationSpec", "ExplanationType", "Renderer", "Segment",
    "parse_document", "score_segment", "is_worth_dynamicizing",
    "classify_segment", "route_renderer", "generate_spec",
]
