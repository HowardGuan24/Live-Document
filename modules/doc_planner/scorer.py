"""
动态化评分器 — 判断"是否值得动态化"

评分维度（各 0-20 分，总分 0-100）：
1. has_formula    — 包含数学符号 / 公式
2. has_process    — 描述步骤 / 流程
3. has_data       — 包含数值 / 数据 / 比较
4. has_spatial     — 描述空间 / 位置 / 变换
5. is_dynamic     — 动作性动词 / 变化描述

阈值：>= 30 分认为"值得动态化"

注意：中文关键词不能用 \b 边界（Python re 对中文 \b 行为异常），
改用中文标点和空白作为天然分隔，直接匹配即可。
"""

import re
from typing import Tuple
from .models import Segment


# --- 关键词 / 模式库 ---
# 中文模式不加 \b，英文模式保留 \b

FORMULA_PATTERNS = [
    r"[=≥≤≠≈±∞∑∏∫√∂∇]",
    r"[αβγδεζηθλμσφψω]",  # 希腊字母
    r"[A-Z]\s*[\+\-\*/\^]\s*[A-Z]",  # 变量间运算：A + B
    r"_\{[^}]+\}",  # 下标：m_{t-1}
    r"\b(?:formula|equation|derivative|gradient|matrix|vector|attention|softmax)\b",
    # 中文数学关键词（不加 \b）
    r"公式|等式|方程|微分|积分|导数|梯度|矩阵|向量|注意力权重|损失函数|正则化|卷积核",
]

PROCESS_PATTERNS = [
    # 中文步骤词（不加 \b）
    r"首先|然后|接着|最后|步骤|流程|阶段|门",
    r"第[一二三四五六七八九十\d]",
    r"\b(?:first|then|next|finally|step|phase|stage|process)\b",
    r"→|->|=>|⇒",
    r"\d+[\.\)、]",
]

DATA_PATTERNS = [
    r"\b\d+[\.\d]*\s*(%|％|倍|万|亿|ms|s|GB|MB|KB|Hz|MHz|GHz|FPS)\b",
    r"\d+[\.\d]*[%％]",  # 百分比数字
    r"数据|统计|对比|比较|百分比|比例|增长率|效率|准确率|延迟|吞吐量|加速比|Top-?1|基线",
    r"\b(?:data|statistics|compare|ratio|percentage|efficiency|accuracy)\b",
    r"\|.*\|.*\|",  # 表格行
]

SPATIAL_PATTERNS = [
    r"位置|方向|坐标|上方|下方|左侧|右侧|中心|边缘|移动|平移|旋转|缩放",
    r"空间|出发|到达|原点|轴|沿|至|距离|角度",
    r"\b(?:position|direction|coordinate|above|below|left|right|center|move|rotate|scale)\b",
    r"曲线|图形|节点|连线|箭头|路径",
    r"\b(?:curve|graph|node|edge|arrow|path)\b",
]

DYNAMIC_PATTERNS = [
    r"变化|演变|递增|递减|趋近|收敛|发散|迭代|循环|反复|逐步|逐渐",
    r"\b(?:change|evolve|increase|decrease|converge|diverge|iterate|loop|repeat|gradually)\b",
    r"更新|下降|上升|传播|扩散|流动|迁移|传递|消失|控制|解决|防止|计算|运行|估计|提取|降维|映射|得到",
    r"\b(?:update|decrease|rise|propagate|diffuse|flow|migrate|transfer)\b",
]


def _score_dimension(text: str, patterns: list) -> int:
    """对一个维度打分 (0-20)"""
    score = 0
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            # 有匹配就给分：每匹配1次 +8，单模式上限 16
            score += min(len(matches) * 8, 16)
    return min(score, 20)


def score_segment(segment: Segment) -> Tuple[float, dict]:
    """
    对一个段落打分。
    返回 (总分 0-100, 各维度分数字典)
    """
    text = segment.text

    dims = {
        "has_formula": _score_dimension(text, FORMULA_PATTERNS),
        "has_process": _score_dimension(text, PROCESS_PATTERNS),
        "has_data": _score_dimension(text, DATA_PATTERNS),
        "has_spatial": _score_dimension(text, SPATIAL_PATTERNS),
        "is_dynamic": _score_dimension(text, DYNAMIC_PATTERNS),
    }

    total = sum(dims.values())
    return total, dims


# 阈值
THRESHOLD = 20


def is_worth_dynamicizing(segment: Segment) -> Tuple[bool, float, dict]:
    """
    判断段落是否值得动态化。
    返回 (是否值得, 总分, 各维度分数)
    """
    total, dims = score_segment(segment)
    return total >= THRESHOLD, total, dims
