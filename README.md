# Live-Document

Live-Document 将技术或教学文档中的知识片段转换为动态解释。目前仓库包含三条可以独立运行的能力链路：

```text
文档内容
  → 文档理解与动画规划（ExplanationSpec）
  → 确定性动画（DSL → Manim → MP4/GIF）
  → 概率生成动画（LTX/Wan → MP4/WebM/GIF）
```

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

## 确定性动画引擎

`modules/animation_engine` 实现了完整的 `JSON → DSL 校验 → Manim → MP4 → GIF` 流程。JSON 不会被转换成任意 Python 执行，而是由固定解释器映射到白名单 Manim 对象和动作。

### 工作流

```text
动画 DSL / 旧版 ExplanationSpec
  ↓
输入归一化与引用校验
  ↓
对象工厂创建 Manim Mobject
  ↓
动作解释器执行 timeline
  ↓
Manim 输出 MP4
  ↓
FFmpeg 调色板优化
  ↓
GIF + normalized_spec.json + result.json
```

### 安装与运行

```bash
# 安装 pytest、Manim 和内置 FFmpeg 运行包
python -m pip install -r requirements.txt

# 仅校验 DSL，不启动渲染
python -m modules.animation_engine \
  modules/animation_engine/examples/gradient_descent.json \
  --validate-only

# 生成梯度下降 MP4 和 GIF
python -m modules.animation_engine \
  modules/animation_engine/examples/gradient_descent.json \
  -o outputs

# 生成 Transformer 数据流 MP4 和 GIF
python -m modules.animation_engine \
  modules/animation_engine/examples/data_flow.json \
  -o outputs
```

Windows PowerShell 同样可以直接执行以上命令；多行命令可改写为单行，或者将 `\` 替换为 PowerShell 续行符。

每个任务写入独立目录：

```text
outputs/<animation-id>/
├── animation.mp4
├── animation.gif
├── normalized_spec.json
└── result.json
```

### DSL能力

当前支持的对象：

- `text`、`formula`、`circle`、`rectangle`、`point`
- `axes`、`graph`、`arrow`、`line`、`group`、`image`

当前支持的动作：

- `create`、`write`、`fade_in`、`fade_out`
- `move`、`move_by`、`follow_path`、`transform`
- `highlight`、`change_color`、`scale`、`rotate`
- `grow_arrow`、`add`、`wait`

时间线支持顺序动作和 `parallel` 并行动作。对象 ID、依赖关系、坐标、动作参数、输出尺寸和帧率都会在渲染前验证。

对于旧版 `ExplanationSpec` 中的字符串 `objects` 和 `steps`，引擎会生成通用步骤动画作为兼容兜底。需要精确控制曲线、节点、箭头和运动轨迹时，应使用显式的 DSL `objects` 与 `timeline`。

### 当前示例与结果

- 梯度下降：损失曲线、参数点、负梯度箭头和逐步收敛，输出为 768×432、约 6.5 秒、97 帧 GIF。
- Transformer 数据流：输入、编码器和输出节点，以及数据标记沿箭头移动，输出为 768×432、108 帧 GIF。
- FFmpeg 优先使用系统 `PATH`，找不到时自动使用 `imageio-ffmpeg` 提供的二进制文件。
- 普通图形和文字动画不要求 LaTeX；`formula` 对象以及开启坐标轴数值标签时需要可用的 LaTeX 环境。

### 测试

```bash
python -m pytest -q
```

当前全仓测试结果为 `26 passed, 3 skipped`，其中确定性动画引擎包含真实的 JSON → Manim → MP4 → GIF 集成测试。

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
│   ├── animation_engine/            # 确定性动画：DSL → Manim → GIF
│   │   ├── schema.py                # DSL 归一化与校验
│   │   ├── manim_renderer.py        # 对象工厂与动作解释器
│   │   ├── media.py                 # FFmpeg GIF 转码与产物检查
│   │   ├── pipeline.py              # 端到端生成流程
│   │   ├── cli.py                   # 命令行入口
│   │   └── examples/                # 梯度下降与数据流示例
│   └── video_model/                # 模块 B：概率生成（视频模型 → GIF）
├── tests/
│   ├── test_acceptance.py           # 文档规划验收测试
│   └── test_animation_engine.py     # 动画引擎单元与集成测试
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
