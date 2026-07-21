"""
生成器 — 构造 ExplanationSpec

职责：
1. 接收已分类、已路由的段落信息
2. 推断 goal / objects / steps
3. 输出完整的 ExplanationSpec
4. 失败时返回 fallback（不适合生成），而不是硬生成
"""

import re
from typing import List, Optional
from .models import (
    Segment, ExplanationType, Renderer, ExplanationSpec,
)
from .scorer import score_segment
from .classifier import classify_segment
from .router import route_renderer


# --- objects 推断关键词 → 对象名 ---

OBJECT_KEYWORDS = {
    # 公式类
    r"梯度|gradient": "gradient_arrow",
    r"损失|loss": "loss_curve",
    r"参数|parameter": "parameter_point",
    r"学习率|learning.?rate": "learning_rate",
    r"权重|weight": "weight_node",
    r"偏置|bias": "bias_value",
    r"激活|activation": "activation_function",
    r"矩阵|matrix": "matrix_grid",
    r"向量|vector": "vector_arrow",
    # 流程类
    r"输入|input": "input_node",
    r"输出|output": "output_node",
    r"层|layer": "layer_box",
    r"节点|node": "node_circle",
    r"边|edge|连接": "edge_line",
    r"数据|data": "data_block",
    # 空间类
    r"曲线|curve": "curve_line",
    r"点|point": "point_dot",
    r"箭头|arrow": "direction_arrow",
    r"坐标|axis": "coordinate_axis",
    r"区域|region": "shaded_area",
}

# --- steps 推断（按类型） ---

PROCESS_STEP_TEMPLATES = {
    ExplanationType.FORMULA: [
        "show_formula", "highlight_variables", "animate_transformation", "show_result"
    ],
    ExplanationType.PROCESS: [
        "show_step_1", "show_step_2", "show_step_3", "show_conclusion"
    ],
    ExplanationType.DATAFLOW: [
        "show_source", "show_transformation", "show_destination", "highlight_flow"
    ],
    ExplanationType.OPERATION: [
        "show_context", "demonstrate_action", "show_effect", "show_summary"
    ],
    ExplanationType.SCENE: [
        "setup_scene", "show_objects", "animate_interaction", "show_outcome"
    ],
}


def _infer_goal(text: str, exp_type: ExplanationType) -> str:
    """根据文本和类型推断动画目标"""
    type_goals = {
        ExplanationType.FORMULA: "直观展示公式的含义与变换过程",
        ExplanationType.PROCESS: "逐步展示流程的各个阶段",
        ExplanationType.DATAFLOW: "展示数据的流动与变换关系",
        ExplanationType.OPERATION: "演示操作步骤与执行效果",
        ExplanationType.SCENE: "构建空间场景并展示动态交互",
    }
    return type_goals.get(exp_type, "展示内容的动态解释")


def _infer_objects(text: str) -> List[str]:
    """从文本中提取可视化对象"""
    objects = []
    for pattern, obj_name in OBJECT_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if obj_name not in objects:
                objects.append(obj_name)
    # 至少返回一个对象
    if not objects:
        objects.append("main_element")
    return objects


def _infer_steps(exp_type: ExplanationType, text: str) -> List[str]:
    """根据类型和文本推断动画步骤"""
    base_steps = PROCESS_STEP_TEMPLATES.get(exp_type, ["show_content"])

    # 对 process 类型，尝试从文本中提取实际步骤数
    if exp_type == ExplanationType.PROCESS:
        step_matches = re.findall(r"(?:步骤|第)[\s]*[一二三四五六七八九十\d]", text)
        if len(step_matches) >= 2:
            steps = [f"show_step_{i+1}" for i in range(len(step_matches))]
            steps.append("show_conclusion")
            return steps

    return base_steps


def generate_spec(
    segment: Segment,
    min_confidence: float = 0.3,
) -> Optional[ExplanationSpec]:
    """
    为一个段落生成 ExplanationSpec。

    返回 None 表示"不适合生成"（分数太低）。
    """
    total, dims = score_segment(segment)
    confidence = min(total / 100.0, 1.0)

    # 分数太低 → 不适合生成
    if total < 30:
        return ExplanationSpec(
            source_text=segment.text,
            type="unsuitable",
            renderer="text_only",
            goal="不适合动态化",
            objects=[],
            steps=[],
            confidence=confidence,
            fallback_reason=f"评分过低 ({total}/100)，内容不适合动态化展示",
        )

    exp_type = classify_segment(segment)
    renderer = route_renderer(exp_type, confidence)

    return ExplanationSpec(
        source_text=segment.text,
        type=exp_type.value,
        renderer=renderer.value,
        goal=_infer_goal(segment.text, exp_type),
        objects=_infer_objects(segment.text),
        steps=_infer_steps(exp_type, segment.text),
        confidence=round(confidence, 2),
        fallback_reason=None,
    )
