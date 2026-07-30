# Stage 2：跨学科 Loop Engineer

Stage 2 把 Stage 1 的三角洲实验扩展为十个跨学科新案例，并用固定协议约束 Agent 的
后续自我迭代。当前 Phase 0–6 已完成并发布 v0.1.0：案例注册、Stage 1 基线、评分协议、回归层级、
单轮预算、五份数据契约、十个 model-free fixture，以及五学科共 245 帧的确定性程序
动画已经机器化并通过验证；15 个图像能力边界实验也已完成盲评与自动检查。Phase 4
已补齐剩余五个案例的 245 张程序帧、冻结十案例数据类型路由，并完成六轮路线实验；
十案例关键帧路线已经稳定；Phase 5 又以 8 次 LTX-2.3 调用和 2 条零模型程序回退
覆盖全部 5 种冻结运动类型。Phase 6 最终又通过十案例合同与关键帧、五学科图片、
五类运动、真实 tokenizer、三角洲历史文件和核心硬编码扫描。

先看：

- `output/phase-6/stage2-final-report.html`：v0.1.0 最终发布报告；从程序状态、
  控制图、完整提示词、raw 供体、受限材质合成、视频与 tokenizer 到失败修正；
- `output/phase-3/report.html`：五学科 15 轮的控制图、提示词、全部 raw、
  composite、失败证据与通用路由表；
- `output/phase-4/program-report.html`：新增五案例程序、十案例路由、GEO-01 与
  BIO-02 路线冒烟，以及程序针孔、CHEM-02 安全但无效的两类失败证据；
- `output/phase-5/report.html`：首尾帧视频、9 个时间点、完整提示词、模型工作流、
  固定锚点与运动方向审计，以及五类运动覆盖状态；
- `output/phase-2/report.html`：五个真实程序动画、关键帧、语义层和机制验收；
- `output/phase-2/sentinel-keyframes.jpg`：五案例各四个关键帧的总览；
- `output/phase-1/report.html`：Phase 1 的十案例数据契约与逐项可视证据；
- `output/phase-1/fixture-contact-sheet.jpg`：十个抽象 fixture 的总览；
- `output/phase-0/report.html`：Phase 0 的基线、评分与预算；
- `case.txt`：十个案例的教学片段、模型职责和案例硬门禁；
- `loop.md`：完整迭代与晋级流程；
- `case_registry.json`：程序读取的稳定案例 ID 和能力标签。

从仓库根目录复现 Phase 0 与 Phase 1：

```bash
.venv/bin/python -m modules.video_model.stage2.phase0
.venv/bin/python -m modules.video_model.stage2.phase0 --check
.venv/bin/python -m modules.video_model.stage2.phase1
.venv/bin/python -m modules.video_model.stage2.phase1 --check
.venv/bin/python -m modules.video_model.stage2.phase2
.venv/bin/python -m modules.video_model.stage2.phase2 --check
.venv/bin/python -m modules.video_model.stage2.phase3 --report
.venv/bin/python -m modules.video_model.stage2.phase3 --check
.venv/bin/python -m modules.video_model.stage2.phase4
.venv/bin/python -m modules.video_model.stage2.phase4 --check
.venv/bin/python -m modules.video_model.stage2.phase5
.venv/bin/python -m modules.video_model.stage2.phase5 --check
/workspace/comfyui-rocm-env/bin/python -m modules.video_model.stage2.phase6_token_audit
.venv/bin/python -m modules.video_model.stage2.phase6_image_regression
.venv/bin/python -m modules.video_model.stage2.phase6
.venv/bin/python -m modules.video_model.stage2.phase6 --check
.venv/bin/python -m pytest -q modules/video_model/stage2/tests
```

Phase 0 命令验证评分协议和 Stage 1 文件哈希；Phase 1 命令生成或复核十个 fixture、
44 个语义层、联系表、报告和 manifest；Phase 2 通过同一运行器生成五个真实程序、
245 帧、20 个关键帧、语义层、MP4 和机制证据。Phase 3 汇总 15 个单变量实验、
112 张 raw 候选、模型指纹、盲评、受限材质 composite 和能力边界。Phase 4 当前新增
245 张程序帧和 24 次图片调用；视频模型调用数仍为 0。Phase 4 中 GEO-01、BIO-02、
MATH-01 与 PHYS-02 各有受限材质候选，CHEM-02 的安全但无效模型结果被拒绝后改用
确定性晶面着色。

Phase 5 使用已部署的 LTX-2.3 22B FLF2V 工作流。连续场传播、对象分裂和边界拓扑
变化由模型通过；精确拼图刚体身份与液体浓度扩散在各两次预算内失败，自动改用程序
轨迹和程序软标量场。液体 evaluator 记录颜色面积与积分色差，拓扑 evaluator 直接
从输出像素反算连通域；后者还发现并修正了 GEO-01 “状态称已隔离、raster 仍连通”
的程序错误。报告保留 ComfyUI 未暴露实际 tokenizer 计数这一发布阻塞，不把字符长度
冒充 token 完整性证据。

Phase 6 已解决该历史阻塞：它从登记的 Gemma safetensors 中读取内嵌
`spiece_model`，调用 ComfyUI 的实际 tokenizer 复核八次模型视频提示词。最终发布
复用 Phase 3 raw 图片供体和 Phase 5 视频证据，没有为封版再抽新种子；三类新增
四帧图片回归使用同一个按语义层保护的多供体材质核心。版本号见 `VERSION`，变更和
已知边界见 `CHANGELOG.md`。
