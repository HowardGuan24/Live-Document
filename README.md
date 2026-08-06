# Live-Document

将技术或教学文档中的概念转换为动态教学视频。仓库以 `modules/` 中的三阶段流程为核心制作管线，`web/` 为产品界面。

## 核心制作流程

`modules/` 中的核心流程把用户概念或教学流程制作成带字幕、可发布的动态教学视频：

```text
用户概念或教学流程
  → Phase 1：程序教学视频与 Bridge
  → Phase 2：真实感关键帧
  → Phase 3：真实感连续视频与最终合成
```

权威规范与详细用法见 [`modules/README.md`](modules/README.md)。

一条命令运行完整流程：

```bash
cd modules
python3 run_pipeline.py --request REQUEST.example.md --run-id seed-demo
```

常用控制参数：`--quality smoke`、`--target realistic`、`--stop-after phase1|phase2`、`--resume --run-id <id>`、`--dry-run`。所有 GPU 任务同时遵循 [`GPU_GENERATION_POLICY.md`](modules/GPU_GENERATION_POLICY.md)。

## 产品界面：web/

`web/` 是前后端分离的产品界面（React + Vite 前端、FastAPI 后端），提供登录鉴权、文档规划、动画任务管理与一键公网部署。详见 [`web/README.md`](web/README.md)。

## 目录结构

```text
├── README.md
├── modules/
│   ├── phase1/                     # 程序教学视频与 Bridge
│   ├── phase2/                     # 真实感关键帧
│   ├── phase3/                     # 连续视频与最终合成
│   └── run_pipeline.py             # 三阶段总入口
└── web/                            # 产品界面（React + FastAPI）
```
