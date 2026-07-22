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
python main.py --text "梯度下降沿负梯度方向更新参数" --pretty

# 处理文件
python main.py data/test_paragraphs.json -o output.json -p

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
├── main.py                         # CLI 入口
├── competition.md                  # 比赛规则
├── requirements.txt                # 环境依赖
├── data/
│   └── test_paragraphs.json        # 20 段验收测试数据
├── modules/
│   ├── doc_planner/                # 模块 A：文档理解与动画规划
│   │   ├── models.py               # 数据模型
│   │   ├── parser.py               # 文档分段 + 章节上下文
│   │   ├── scorer.py               # 5 维动态化评分器
│   │   ├── classifier.py           # 5 类分类器
│   │   ├── router.py               # 渲染器路由
│   │   └── generator.py            # ExplanationSpec 生成
│   └── video_model/                # 模块 B：概率生成（视频模型 → GIF）
├── tests/
│   └── test_acceptance.py          # 验收测试
├── demo/                           # 概念演示
└── plans/
    └── v1                          # MVP 初始计划
```

<!-- MODULE_B_START -->

## 模块 B：解释型短视频生成

### 当前工作流

`ExplanationSpec JSON → 版本化解释型提示词 → LTX/Wan/程序化后端 → 原始 MP4 → ffmpeg 标准化 → MP4/WebM/GIF → ffprobe、哈希与 V2 metadata`。

### 当前支持的后端

- `ltx`：V2 主后端，LTX-Video 0.9.8 2B distilled，面向 W7900 + ROCm。
- `wan`：Wan2.1-T2V-1.3B 对照基线。
- `procedural`：安装、CI 和故障回退；状态为 `success_fallback`，不会伪装成模型成功。
- `auto`：按 LTX → Wan → procedural 尝试并记录每次失败原因；发生任何后端替换时状态为 `success_fallback`。

### 当前最佳配置

待固定基准完成后填写；在有视觉评测证据前，不宣称 `fast`、`balanced` 或 `quality` 最优。

### Benchmark

已固定 4 个速度案例、8 个质量案例、3 个 profile，并支持冷/热耗时、VRAM、JSONL 与 Markdown 汇总。真实 GPU 结果见 `modules/video_model/benchmarks/results/`。

### 已完成

V2 后端接口、模型复用、原始产物保留、显式回退状态、版本化提示词、多人候选、质量 rubric、AI 导入格式、人工成对 CSV 和单元测试。

### 当前问题

本机首次 6.34 GB LTX 权重下载的可选 Xet 路径返回 HTTP 416；标准分块 HTTP 可继续下载。概率生成的语义质量仍必须由固定 rubric 的 AI/人工评测确认。

### 下一步

完成真实 LTX/Wan 同案例对照，记录受控实验的接受/拒绝结论，再据速度与人工质量选择默认 profile。

<!-- MODULE_B_END -->
