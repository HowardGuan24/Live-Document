"""
生成器 — 构造 LearningSpec

职责：
1. 接收已评分、已分类的段落
2. 提取 learning_goal / entities / state_variables / causal_steps
3. 推断 invariants / comprehension_questions
4. 输出完整的 LearningSpec
5. 失败时返回 fallback（不适合生成），而不是硬生成
"""

import re
from typing import List, Optional
from .models import Segment, LearningSpec
from .scorer import score_segment
from .classifier import classify_segment


# ── 实体关键词 ──────────────────────────────────────────────
# 匹配到即纳入 entities，避免重复

ENTITY_PATTERNS = [
    # 自然/地理
    r"水流|河流|泥沙|沉积|河口|三角洲|海洋|水域|陆地|沙洲|海岸",
    r"风|雨|云|冰|雪|水蒸气|大气|地壳|岩浆|板块",
    r"牛轭湖|曲流|河曲|河道|河岸|洪水|洪水期|堰塞湖|冲积扇",
    # 机器学习
    r"梯度|损失函数|参数|权重|偏置|学习率|激活函数|卷积|池化|注意力",
    r"输入层|输出层|隐藏层|全连接层|卷积层|编码器|解码器",
    # 物理/数学
    r"向量|矩阵|标量|力|速度|加速度|能量|动量|频率|波|光线|磁场|电场",
    # 通用对象
    r"节点|边|曲线|点|箭头|图形|数据|信号|样本|特征|标签",
]

# ── 状态变量关键词 ────────────────────────────────────────────

STATE_VAR_PATTERNS = [
    r"流速|速度|速率|加速度",
    r"位置|坐标|方向|角度",
    r"厚度|大小|范围|面积|体积|长度|宽度|高度",
    r"温度|压力|浓度|密度|湿度",
    r"准确率|损失|误差|概率|权重值",
    r"数量|数目|比例|百分比|率",
    r"距离|半径|直径|周长",
]

# ── 因果关系模式 ──────────────────────────────────────────────

# "因为A，所以B" / "由于A，导致B" / "A，因此B"
CAUSAL_CONNECTIVES = [
    (r"因为(.{2,30}?)[，,](?:所以|因此|故)(.{2,30}?)[。；]", "因为…所以…"),
    (r"由于(.{2,30}?)[，,](?:导致|引起|造成)(.{2,30}?)[。；]", "由于…导致…"),
    (r"当(.{2,30}?)(?:时|时候)[，,](.{2,30}?)[。；]", "当…时，…"),
]

# "首先A，然后B，最后C" 序列
SEQUENCE_PATTERNS = [
    r"首先(.{2,40}?)[，,]",
    r"然后(.{2,40}?)[，,]",
    r"接着(.{2,40}?)[，,]",
    r"最后(.{2,40}?)[。；]",
    r"第[一二三四五六七八九十\d]步[：:,]?\s*(.{2,40}?)[，,。；]",
]

# ── 视觉证据推断 ──────────────────────────────────────────────

VISUAL_HINTS = {
    r"下降|减少|降低|收缩|缩小|变短|变小": "数值变小 / 箭头变短",
    r"上升|增加|增大|膨胀|扩大|变长|变大": "数值变大 / 箭头变长",
    r"停止|静止|固定|不再": "对象停止运动",
    r"移动|流动|迁移|传播|前进": "对象沿路径移动",
    r"沉积|堆积|积累|累积|沉淀": "颗粒沉降并形成新层",
    r"分散|扩散|发散|散开|分裂": "对象向多个方向扩散",
    r"合并|汇聚|聚集|收敛|集中": "多个对象汇聚到一处",
    r"旋转|转动|翻转": "对象绕轴旋转",
    r"出现|形成|产生|生成|创建": "新对象渐入显示",
    r"消失|消散|溶解|消亡": "对象渐出消失",
    r"变换|转化|改变|变成|成为": "对象形态发生变化",
    r"连接|相连|接合|对接": "连线出现或延伸",
    r"断裂|断开|分离|脱离": "连线断裂或对象分离",
    r"升高|升温|变热": "颜色偏暖 / 数值上升",
    r"降低|降温|变冷": "颜色偏冷 / 数值下降",
}

# ── 约束/不变量模式 ──────────────────────────────────────────

INVARIANT_PATTERNS = [
    r"不能(.{2,30}?)[。；]",
    r"必须(.{2,30}?)[。；]",
    r"不允许(.{2,30}?)[。；]",
    r"始终(.{2,30}?)[。；]",
    r"保持(.{2,30}?)[。；]",
    r"总是(.{2,30}?)[。；]",
    r"前提是(.{2,30}?)[。；]",
]

# ── 提取函数 ─────────────────────────────────────────────────

def _extract_entities(text: str) -> List[str]:
    """从文本中提取关键实体"""
    entities = []
    for pat in ENTITY_PATTERNS:
        for match in re.finditer(pat, text):
            e = match.group()
            if e not in entities:
                entities.append(e)
    return entities


def _extract_state_variables(text: str) -> List[str]:
    """从文本中提取状态变量"""
    variables = []
    for pat in STATE_VAR_PATTERNS:
        for match in re.finditer(pat, text):
            v = match.group()
            if v not in variables:
                variables.append(v)
    return variables


def _infer_visual_evidence(change_text: str) -> str:
    """根据变化描述推断视觉证据"""
    for pat, visual in VISUAL_HINTS.items():
        if re.search(pat, change_text):
            return visual
    return "相应对象发生变化"


def _extract_chain(text: str) -> List[str]:
    """提取「从A，到B、C、D，最终E」链式结构，返回按序的阶段列表。

    适用于自然过程/成因描述，例如：
    从逐渐弯曲，到曲流颈部变窄、洪水期河流截弯取直、旧河道被泥沙封堵，
    最终形成牛轭湖 → ["逐渐弯曲", "曲流颈部变窄", "洪水期河流截弯取直",
    "旧河道被泥沙封堵", "形成牛轭湖"]
    """
    m = re.search(r"从(.+?)[，,]\s*(?:到|直至|逐渐)(.+?)(?:最终|最后|随后|终于)(.+?)(?:[。；]|$)", text)
    if not m:
        return []
    start = m.group(1).strip().rstrip("的")
    middle = m.group(2).strip()
    end = m.group(3).strip().rstrip("的完整过程").rstrip("的过程")
    stages: List[str] = []
    if start:
        stages.append(start)
    # 中间可含多个「、」分隔的阶段
    for part in re.split(r"[、，,]", middle):
        part = part.strip()
        if part:
            stages.append(part)
    if end:
        stages.append(end)
    return stages


def _extract_causal_steps(text: str) -> List[dict]:
    """
    从文本中提取因果步骤链。
    优先用因果连接词，退化到步骤序列。
    """
    steps = []

    # 尝试因果连接词
    for pat, _ in CAUSAL_CONNECTIVES:
        for match in re.finditer(pat, text):
            cause = match.group(1).strip()
            change = match.group(2).strip()
            steps.append({
                "cause": cause,
                "change": change,
                "visual_evidence": _infer_visual_evidence(change),
            })

    if steps:
        return steps

    # 退化：用步骤序列（首先/然后/最后/第X步）
    seq_matches = []
    for pat in SEQUENCE_PATTERNS:
        for match in re.finditer(pat, text):
            seq_matches.append(match.group(1).strip() if match.lastindex else match.group().strip())

    if len(seq_matches) >= 2:
        for i, action in enumerate(seq_matches):
            cause = seq_matches[i - 1] if i > 0 else "起始状态"
            steps.append({
                "cause": cause,
                "change": action,
                "visual_evidence": _infer_visual_evidence(action),
            })
        return steps

    # 退化：从X，到Y、Z、W，最终V 的链式结构（自然过程/成因描述）
    # 例：从逐渐弯曲，到颈部变窄、洪水期截弯取直、旧河道被泥沙封堵，最终形成牛轭湖
    chain = _extract_chain(text)
    if chain:
        for i, change in enumerate(chain):
            cause = chain[i - 1] if i > 0 else "起始状态"
            steps.append({
                "cause": cause,
                "change": change,
                "visual_evidence": _infer_visual_evidence(change),
            })
        return steps

    # 再退化：拆分句号分隔的句子，相邻两句构成因果
    sentences = re.split(r"[。；]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]
    if len(sentences) >= 2:
        for i in range(1, len(sentences)):
            steps.append({
                "cause": sentences[i - 1],
                "change": sentences[i],
                "visual_evidence": _infer_visual_evidence(sentences[i]),
            })
        return steps

    return steps


def _extract_invariants(text: str) -> List[str]:
    """提取约束/不变量"""
    invariants = []
    for pat in INVARIANT_PATTERNS:
        for match in re.finditer(pat, text):
            invariants.append(match.group().strip())
    return invariants


def _generate_questions(text: str, entities: List[str], learning_goal: str) -> List[str]:
    """根据内容生成理解检测问题"""
    questions = []

    # 基于实体生成 "为什么" 问题
    if len(entities) >= 2:
        questions.append(f"{entities[0]}为什么影响{entities[1]}？")
    if entities:
        questions.append(f"{entities[0]}是如何变化的？")

    # 基于学习目标生成
    if learning_goal and "理解" in learning_goal:
        topic = learning_goal.replace("理解", "").strip()
        if topic:
            questions.append(f"{topic}的关键原因是什么？")

    return questions


def _infer_learning_goal(text: str, section: str, exp_type) -> str:
    """推断学习主旨"""
    # 如果有章节标题，优先用
    if section and len(section) > 2:
        return f"理解{section}的核心机制"

    # 按类型推断
    type_goals = {
        "formula": "理解公式的含义与变换过程",
        "process": "理解流程的各阶段及其因果关系",
        "dataflow": "理解数据的流动与变换关系",
        "operation": "理解操作步骤与执行效果",
        "scene": "理解空间场景中的动态交互",
    }
    return type_goals.get(exp_type, "理解该内容的动态变化过程")


# ── 主入口 ────────────────────────────────────────────────────

def generate_spec(segment: Segment) -> Optional[LearningSpec]:
    """
    为一个段落生成 LearningSpec。

    评分低于阈值时返回 fallback（不适合动态化），不硬生成。
    """
    total, _ = score_segment(segment)

    # 分数太低 → fallback
    if total < 20:
        return LearningSpec(
            learning_goal=None,
            entities=[],
            state_variables=[],
            causal_steps=[],
            invariants=[],
            comprehension_questions=[],
            fallback_reason=f"评分过低 ({total}/100)，内容不适合动态化展示",
        )

    exp_type = classify_segment(segment)
    text = segment.text

    entities = _extract_entities(text)
    state_variables = _extract_state_variables(text)
    causal_steps = _extract_causal_steps(text)
    invariants = _extract_invariants(text)
    learning_goal = _infer_learning_goal(text, segment.section, exp_type.value)
    questions = _generate_questions(text, entities, learning_goal)

    return LearningSpec(
        learning_goal=learning_goal,
        entities=entities,
        state_variables=state_variables,
        causal_steps=causal_steps,
        invariants=invariants,
        comprehension_questions=questions,
        fallback_reason=None,
    )
