# Live Document 核心制作流程

`modules/` 直接包含一条三阶段制作流程：

```text
用户概念或教学流程
        ↓
Phase 1：程序教学视频与 Bridge
        ↓
Phase 2：真实感关键帧
        ↓
Phase 3：真实感连续视频与最终合成
```

统一使用 `Phase 1`、`Phase 2`、`Phase 3` 命名。新的基础文件、脚本和文档不得再使用旧的阶段命名。所有 GPU 图片和视频任务同时遵循 [`GPU_GENERATION_POLICY.md`](GPU_GENERATION_POLICY.md)。

## 一条命令运行完整流程

把自然语言请求保存为 Markdown，然后运行：

```bash
python3 run_pipeline.py --request REQUEST.example.md --run-id seed-demo
```

也可以直接传入短文本或从标准输入读取：

```bash
python3 run_pipeline.py --text "种子是怎样萌发并成长为幼苗的？"
python3 run_pipeline.py --request - --run-id seed-demo
```

默认使用 `release` 质量并根据 Phase 1 route 自动选择路径。常用控制参数：

- `--quality smoke`：使用保守规格跑通所有必要片段并生成最终视频；
- `--target realistic`：若 Phase 1 判定不适合真实化则明确失败；
- `--stop-after phase1|phase2`：在阶段边界停止；
- `--resume --run-id <id>`：使用原输入和原配置继续失败或主动停止的 run；
- `--dry-run`：只检查输入并显示预定路径，不启动 Agent 或 GPU。

总控状态位于 `runs/<run-id>/pipeline.json`，里程碑位于 `events.jsonl`，最终交付为 `runs/<run-id>/final_video.mp4`。默认不会读取其他历史 runs。

## 目录

```text
modules/
├── README.md
├── AGENTS.md
├── PIPELINE_PROMPT.md
├── GPU_GENERATION_POLICY.md
├── REQUEST.example.md
├── run_pipeline.py
├── runs/
├── phase1/
│   ├── PHASE1_PROMPT.md
│   ├── AGENTS.md
│   ├── README.md
│   ├── REQUEST.example.md
│   ├── run_phase1.sh
│   ├── tools/
│   └── runs/
├── phase2/
│   ├── PHASE2_PROMPT.md
│   ├── tools/
│   └── runs/
└── phase3/
    ├── PHASE3_PROMPT.md
    ├── tools/
    └── runs/
```

## Phase 1：程序教学视频

Phase 1 把用户提供的概念、粗略流程或详细课本文档转换为确定性程序动画，并定义后续真实化需要的关键状态。

权威规范：[`phase1/PHASE1_PROMPT.md`](phase1/PHASE1_PROMPT.md)

### 运行

```bash
cd phase1
cp REQUEST.example.md REQUEST.md
# 编辑 REQUEST.md
./run_phase1.sh REQUEST.md <run-id>
```

`run-id` 可省略；省略时自动使用时间戳。

### 输出

```text
phase1/runs/<run-id>/
├── REQUEST.md
├── brief.md
├── app/
├── subtitles.srt
├── video.mp4
├── poster.png
├── agent-final.txt
└── bridge/
    ├── manifest.json
    ├── contact_sheet.png
    ├── presentation/
    ├── clean/
    └── overlay/
```

Phase 1 决定：

- 教学主线和时间顺序；
- 程序画面如何连续演化；
- 关键对象、状态和事件；
- 事件发生前后的关键时刻；
- 哪些时刻适合进入 Phase 2。

`bridge/manifest.json` 必须由导出工具从程序的 `LIVE_DOCUMENT_BRIDGE` 生成，不能手写。

## Phase 2：真实感关键帧

Phase 2 从 Phase 1 的 Bridge 中选择少量必要状态，将其转换为同一世界、同一视角下的真实感关键帧。

权威规范：[`phase2/PHASE2_PROMPT.md`](phase2/PHASE2_PROMPT.md)

Phase 2 当前使用 Fast Mode，目标是在有限时间内输出 4～5 张可用于后续视频化的 anchors。

### 调用方式

向 Agent 提供 Phase 1 run 和新的 Phase 2 输出目录，例如：

```text
遵循 phase2/PHASE2_PROMPT.md 执行 Phase 2。

Phase 1 run：phase1/runs/<run-id>
Phase 2 输出：phase2/runs/<phase2-run-id>

使用现有 FLUX.2 / ComfyUI 环境，不修改 Phase 1。
```

### 输入

Phase 2 默认只读取：

```text
phase1/runs/<run-id>/brief.md
phase1/runs/<run-id>/bridge/manifest.json
phase1/runs/<run-id>/bridge/clean/
```

只有关键状态含义不清楚时，才额外检查 presentation、overlay、视频或 `renderFrame(t)`。

### 输出

```text
phase2/runs/<phase2-run-id>/
├── selected_anchors.json
├── world_reference.png
├── contact_sheet.png
├── report.md
└── anchors/
    └── <anchor-id>/
        ├── input_clean.png
        ├── prompt.txt
        └── realistic.png
```

Phase 2 决定：

- 从 Phase 1 候选中选择哪 4～5 个 anchors；
- 哪张稳定帧用于建立 world reference；
- 如何把程序材质自然化；
- 哪些失败帧需要进行唯一一次重试；
- 哪些已知问题可以带入后续视频化试跑。

## Phase 1 → Phase 2 交接

Bridge 是两阶段之间的唯一正式交接层。

| Phase 1 产物 | Phase 2 用途 |
|---|---|
| `brief.md` | 理解完整教学主线 |
| `manifest.route` | 判断是否进入真实化 |
| `manifest.keyMoments` | generation anchor 候选 |
| `manifest.events` | 理解事件前后关系 |
| `description` / `preserve` | 编写当前状态 prompt |
| `worldContinuity` | 维持环境和对象身份 |
| `clean/<id>.png` | 当前 anchor 的结构输入 |
| `presentation/` / `overlay/` | 仅在语义不清时辅助判断 |

Route 规则：

- `programmatic`：通常停在 Phase 1；
- `realizable`：Phase 2 可以从全部 key moments 中选择；
- `hybrid`：Phase 2 只选择 `realizable: true` 的 key moments。

Phase 2 可以减少 anchors，但不能改变 Phase 1 的教学顺序、对象身份或事件定义。Phase 1 决定“发生什么”，Phase 2 决定“真实世界中看起来怎样”。

## Phase 3：连续视频与最终合成

Phase 3 使用 Phase 1 的事件定义控制运动过程，使用 Phase 2 的真实关键帧约束相邻片段的首尾状态。

权威规范：[`phase3/PHASE3_PROMPT.md`](phase3/PHASE3_PROMPT.md)

Phase 3 按相邻 anchors 分段生成，先验证一个代表性 smoke，再串行生成其余片段。它负责去除重复端帧、拼接 `base_video.mp4`，并重新映射 Phase 1 的 overlay 与字幕生成 `final_video.mp4`。Phase 1 和 Phase 2 的来源产物保持只读。

## 命名约定

- 阶段目录：`phase1/`、`phase2/`、`phase3/`
- 权威 prompt：`PHASE1_PROMPT.md`、`PHASE2_PROMPT.md`、`PHASE3_PROMPT.md`
- Phase 1 运行脚本：`run_phase1.sh`
- 运行产物：`phase1/runs/<run-id>/`、`phase2/runs/<run-id>/`、`phase3/runs/<run-id>/`
- 新文档和日志统一写 `Phase 1`、`Phase 2`、`Phase 3`
- 历史 runs 是既有产物，不为了术语统一批量改写
