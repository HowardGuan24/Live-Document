"""
路由器 — 根据解释类型选择渲染器

路由规则（确定性优先）：
- formula   → manim（数学动画最佳）
- process   → css_animation（步骤流适合 Web 动画）
- dataflow  → svg（数据流图适合矢量渲染）
- operation → css_animation（操作演示适合 Web）
- scene     → three_js（空间场景适合 3D）

当类型模糊时，用置信度决定是否降级到 text_only。
"""

from .models import ExplanationType, Renderer


# 确定性路由表
ROUTE_TABLE = {
    ExplanationType.FORMULA: Renderer.MANIM,
    ExplanationType.PROCESS: Renderer.CSS_ANIMATION,
    ExplanationType.DATAFLOW: Renderer.SVG,
    ExplanationType.OPERATION: Renderer.CSS_ANIMATION,
    ExplanationType.SCENE: Renderer.THREE_JS,
}


def route_renderer(exp_type: ExplanationType, confidence: float = 1.0) -> Renderer:
    """
    根据解释类型和置信度选择渲染器。

    - confidence >= 0.5 → 确定性路由
    - confidence < 0.5  → 降级到 text_only
    """
    if confidence < 0.5:
        return Renderer.TEXT_ONLY
    return ROUTE_TABLE.get(exp_type, Renderer.TEXT_ONLY)
