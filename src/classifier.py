"""
分类器 — 将段落分类为五种解释类型

分类逻辑（按优先级）：
1. FORMULA  — 数学符号/公式占比最高
2. PROCESS  — 步骤/流程关键词最多
3. DATAFLOW — 数据/数值/比较最突出
4. OPERATION — 操作指令/动作步骤最突出
5. SCENE    — 空间/视觉描述最突出（兜底）

当多个维度分数相同时，按以上优先级取最高。
"""

from .models import Segment, ExplanationType
from .scorer import score_segment


def classify_segment(segment: Segment) -> ExplanationType:
    """
    对一个已通过评分阈值的段落进行分类。
    """
    _, dims = score_segment(segment)

    # 按优先级定义映射
    priority_order = [
        ("has_formula", ExplanationType.FORMULA),
        ("has_process", ExplanationType.PROCESS),
        ("has_data", ExplanationType.DATAFLOW),
        ("has_spatial", ExplanationType.SCENE),
        ("is_dynamic", ExplanationType.OPERATION),
    ]

    # 找出得分最高的维度（相同分数取优先级高的）
    best_type = ExplanationType.SCENE  # 兜底
    best_score = -1

    for dim_key, exp_type in priority_order:
        if dims[dim_key] > best_score:
            best_score = dims[dim_key]
            best_type = exp_type

    return best_type
