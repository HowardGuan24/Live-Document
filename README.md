# Live-Document

将技术或教学文档中的概念转换为动态教学视频。仓库以 `modules/video_model/Re_0` 为核心制作管线，`web/` 为产品界面；`doc_planner` 与 `animation_engine` 作为 web 后端的依赖保留。

## 核心制作流程：Re_0

`modules/video_model/Re_0` 是一条三阶段制作流程，把用户概念或教学流程制作成带字幕、可发布的动态教学视频：

```text
用户概念或教学流程
  → Phase 1：程序教学视频与 Bridge
  → Phase 2：真实感关键帧
  → Phase 3：真实感连续视频与最终合成
```

权威规范与详细用法见 [`modules/video_model/Re_0/README.md`](modules/video_model/Re_0/README.md)。

一条命令运行完整流程：

```bash
cd modules/video_model/Re_0
python3 run_pipeline.py --request REQUEST.example.md --run-id seed-demo
```

常用控制参数：`--quality smoke`、`--target realistic`、`--stop-after phase1|phase2`、`--resume --run-id <id>`、`--dry-run`。所有 GPU 任务同时遵循 [`GPU_GENERATION_POLICY.md`](modules/video_model/Re_0/GPU_GENERATION_POLICY.md)。

## 产品界面：web/

`web/` 是前后端分离的产品界面（React + Vite 前端、FastAPI 后端），提供登录鉴权、文档规划、动画任务管理与一键公网部署。详见 [`web/README.md`](web/README.md)。

## 支撑模块（web 后端依赖）

以下两个模块由 `web/backend` 直接 import，作为产品后端的功能支撑，因此保留在仓库中：

### 文档规划：`modules/doc_planner`

从文档到 `LearningSpec` 的结构化规划管线：`parser`（分段）→ `scorer`（5 维动态化评分）→ `classifier`（5 类）→ `router`（渲染器路由）→ `generator`（LearningSpec JSON）。可直接经顶层 CLI 使用：

```bash
python main.py --text "梯度下降沿负梯度方向更新参数" --pretty
python main.py data/test_paragraphs.json -o output.json -p
```

### 确定性动画：`modules/animation_engine`

`JSON → DSL 校验 → Manim → MP4 → GIF` 的确定性渲染引擎，JSON 由固定解释器映射到白名单 Manim 对象与动作，不执行任意 Python。示例与 DSL 能力见模块目录。

```bash
python -m modules.animation_engine modules/animation_engine/examples/gradient_descent.json -o outputs
```

## 目录结构

```text
├── main.py                         # doc_planner CLI 入口
├── competition.md                  # 比赛规则
├── requirements.txt                # 环境依赖（pytest / manim）
├── data/
│   └── test_paragraphs.json        # 验收测试数据
├── modules/
│   ├── doc_planner/                # 文档规划 → LearningSpec（web 依赖）
│   ├── animation_engine/           # 确定性动画 DSL → Manim → GIF（web 依赖）
│   └── video_model/
│       └── Re_0/                   # 核心制作管线（三阶段）
├── tests/                          # doc_planner / animation_engine 测试
└── web/                            # 产品界面（React + FastAPI）
```

## 测试

```bash
python -m pytest -q
```
