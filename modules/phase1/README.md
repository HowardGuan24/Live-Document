# Live Document Phase 1

Phase 1 使用一个无记忆 Codex Agent，在一次运行中完成：

```text
理解 → 必要补全 → 简短规划 → 编码 → 渲染 → 检查 → 修正
```

输入可以是概念、粗略流程或详细流程。输出是带画面内字幕的程序视频、源码和独立 SRT 字幕。

## 目录

```text
phase1/
├── AGENTS.md
├── PHASE1_PROMPT.md
├── REQUEST.example.md
├── run_phase1.sh
├── package.json
├── tools/
│   ├── render_video.mjs
│   ├── export_bridge.mjs
│   ├── validate_outputs.py
│   └── validate_bridge.py
├── examples/
│   ├── concept.md
│   ├── outline.md
│   └── detailed_process.md
└── runs/
```

一次运行只产生：

```text
runs/<run-id>/
├── REQUEST.md
├── brief.md
├── app/
├── subtitles.srt
├── video.mp4
├── poster.png
├── bridge/
│   ├── manifest.json
│   ├── contact_sheet.png
│   ├── presentation/
│   ├── clean/
│   └── overlay/
└── agent-final.txt
```

`programmatic` route 只要求 `bridge/manifest.json`，不一定生成图片目录和
`contact_sheet.png`。

## 环境

需要：

- Codex CLI
- Node.js 18+
- FFmpeg 与 ffprobe
- Playwright Chromium

在 `phase1` 下安装一次渲染依赖：

```bash
npm install
npx playwright install chromium
```

## 使用

复制并编辑请求：

```bash
cp REQUEST.example.md REQUEST.md
./run_phase1.sh REQUEST.md
```

也可以运行示例：

```bash
./run_phase1.sh examples/concept.md karst-concept
./run_phase1.sh examples/outline.md karst-outline
./run_phase1.sh examples/detailed_process.md karst-detailed
```

第二个参数是可选的 run id。省略时自动使用时间戳。

脚本使用 `codex exec --ephemeral --sandbox workspace-write`。模型和 reasoning effort 默认继承你的 Codex 配置，因此不会覆盖你已经选择的 GPT-5.6 Sol Extra High。

## 设计原则

Phase 1 只固定“窄腰接口”：

- 输入文件；
- `renderFrame(t, options)`；
- 必要输出文件；
- 渲染器和验证器。

内容范围、场景数、时长、对象数量、具体动画方式由 Agent 根据教学需要自行决定。

## Bridge route

- `programmatic`：程序视觉最适合教学，不要求真实化关键帧资产。
- `realizable`：所有关键时刻都导出 presentation、clean 和 overlay 资产。
- `hybrid`：只为标记为可真实化的关键时刻导出三种资产。

手动重新导出并验证 Bridge：

```bash
node tools/export_bridge.mjs \
  --app runs/<run-id>/app/index.html \
  --output runs/<run-id>/bridge

python3 tools/validate_bridge.py runs/<run-id>
```

Phase 1 Agent 负责定义关键教学时刻。Exporter 不重新选择关键帧或设计分镜，
只把网页已经定义的时刻兑现为后续图像和视频真实化阶段使用的资产。字幕继续由
`subtitles.srt` 管理，不进入 overlay。
