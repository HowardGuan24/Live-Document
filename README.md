# Live-Document

## 模块 A：文档理解与动画规划

从文档到 `ExplanationSpec` 的全流程管线，负责识别值得动态化的内容片段，分类并生成结构化动画计划。

### 管线架构

```text
文档输入
  ↓
parser.py       分段 + 章节上下文提取
  ↓
scorer.py       5 维动态化评分（公式 / 流程 / 数据 / 空间 / 动态）
  ↓ (≥20 分)
classifier.py   分类：formula | process | dataflow | operation | scene
  ↓
router.py       确定性路由 → manim | css_animation | three_js | svg | text_only
  ↓
generator.py    生成 ExplanationSpec（结构化 JSON）
```

### 快速开始

```bash
# 处理单段文本
python -m src.main --text "梯度下降沿负梯度方向更新参数" --pretty

# 处理文件
python -m src.main data/test_paragraphs.json -o output.json -p

# 运行验收测试
python tests/test_acceptance.py
```

### 输出示例

```json
{
  "source_text": "梯度下降沿负梯度方向更新参数",
  "type": "formula",
  "renderer": "manim",
  "goal": "直观展示公式的含义与变换过程",
  "objects": ["gradient_arrow", "parameter_point"],
  "steps": ["show_formula", "highlight_variables", "animate_transformation", "show_result"],
  "confidence": 0.52,
  "fallback_reason": null
}
```

不适合动态化的内容返回 fallback 而非硬生成：

```json
{
  "source_text": "今天的天气很好",
  "type": "unsuitable",
  "renderer": "text_only",
  "fallback_reason": "评分过低 (0/100)，内容不适合动态化展示"
}
```

### 目录结构

```text
├── src/
│   ├── models.py          # 数据模型：Segment、ExplanationSpec、枚举
│   ├── parser.py          # 文档分段 + 章节上下文
│   ├── scorer.py          # 5 维动态化评分器
│   ├── classifier.py      # 5 类分类器
│   ├── router.py          # 类型 → 渲染器路由
│   ├── generator.py       # ExplanationSpec 生成
│   └── main.py            # CLI 入口
├── data/
│   └── test_paragraphs.json   # 20 段验收测试数据
├── tests/
│   └── test_acceptance.py     # 验收测试
├── demo/                      # 概念演示
└── plans/
    └── v1                     # MVP 初始计划
```
