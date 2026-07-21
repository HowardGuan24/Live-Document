# live-document

文档 → ExplanationSpec 全流程管线。

将任意文档拆分为段落，找出值得动态化展示的内容，分类并生成结构化动画计划（JSON），提交给下游渲染器（manim / css_animation / three.js / svg）。

## 快速开始

```bash
# 处理单段文本
python -m src.main --text "梯度下降沿负梯度方向更新参数" --pretty

# 处理文件
python -m src.main data/test_paragraphs.json -o output.json -p

# 运行验收测试
python tests/test_acceptance.py
```

## 管线架构

```
文档输入
  ↓
parser.py       文档分段与上下文提取
  ↓
scorer.py       "是否值得动态化" 评分（0-100）
  ↓ (≥20 分通过)
classifier.py   分类：formula / process / dataflow / operation / scene
  ↓
router.py       路由到渲染器：manim / css_animation / three_js / svg
  ↓
generator.py    生成 ExplanationSpec（结构化 JSON）
  ↓
下游渲染器
```

## 输出示例

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

不适合动态化的内容会返回 fallback：

```json
{
  "source_text": "今天的天气很好",
  "type": "unsuitable",
  "renderer": "text_only",
  "fallback_reason": "评分过低 (0/100)，内容不适合动态化展示"
}
```

## 验收标准

给 20 个测试段落，至少能够：
- ✅ 找出合理的动态化位置（100% 准确率）
- ✅ 大致选对生成方式（81% 准确率）
- ✅ 输出能够被下游程序读取的 JSON
- ✅ 失败时给出"不适合生成"，而不是硬生成

## 目录结构

```
├── README.md
├── competition.md
├── requirements.txt
├── data/
│   └── test_paragraphs.json    # 20 段测试数据
├── src/
│   ├── models.py               # 数据模型
│   ├── parser.py               # 文档分段
│   ├── scorer.py               # 动态化评分
│   ├── classifier.py           # 类型分类
│   ├── router.py               # 渲染器路由
│   ├── generator.py            # ExplanationSpec 生成
│   └── main.py                 # CLI 入口
└── tests/
    └── test_acceptance.py      # 验收测试
```
