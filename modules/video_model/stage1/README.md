# Stage 1 复现说明

本目录把机制状态、验证、渲染和模型增强分开。`causal_delta` 可在 CPU 上完整复现；
`keyframe_render` 只有检测到指定的本地 SDXL/ControlNet 权重与运行时后才会推理，
不会静默替换模型。

## 一键运行 CPU 部分并评估已有模型结果

从仓库根目录执行：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt
.venv/bin/python -m modules.video_model.stage1.run
```

这个命令会重跑 Track A，并准备、评估 Track B 已经存在的候选；它不会自动占用 GPU
生成新的 SDXL 候选。第一次完整复现时，还必须执行下一节的 GPU 命令。

输出位于 `modules/video_model/stage1/output/`。Track A 的正式结果在
`output/causal_delta/`。Track B 只需先看 `output/keyframe_render/final/`：
这里是最终首帧、尾帧和对比图；`review/` 是人工候选对比，`_work/` 是可忽略的
生成中间产物。每类文件的用途和可删除性见 `output/keyframe_render/report.md`。

## 分步运行

```bash
.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export

.venv/bin/python -m modules.video_model.stage1.keyframe_render.prepare
.venv/bin/python -m modules.video_model.stage1.keyframe_render.enhance --status-only
```

当前服务器的 ROCm PyTorch 位于 `/opt/venv`。实际 SDXL 推理应使用：

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.enhance \
  --generate-candidates --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.evaluate
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.enhance \
  --generate-pair --selected-style physical_geography --seed 3102 --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.evaluate
```

Track B 默认查找：

```text
/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0
/workspace/ai-concept-animator/.cache/models/controlnet-canny-sdxl-1.0
```

也可设置 `SDXL_BASE_PATH` 与 `SDXL_CANNY_CONTROLNET_PATH`。设置只改变权重位置，
不会改变固定的历史参数。

算法说明、精确参数、模型哈希、每个中间文件的用途和从零复现步骤分别见：

- `output/causal_delta/report.md`
- `output/keyframe_render/report.md`

若已有外部纹理提案，可独立验证 mask projection，而无需把它冒充为指定模型输出：

```bash
.venv/bin/python -m modules.video_model.stage1.keyframe_render.enhance \
  --proposal proposal.png --keyframe last --candidate-name review-01
```

## Stage 1.1：第一张图路线实验

`stage1.1.md` 比较模型自检、自由俯视、受控俯视和水下场景。它与旧 Track B 分开，
不会执行 semantic mask projection。完整生成与报告命令：

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \
  --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \
  --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.first_frame_test \
  --report
```

结果在 `output/keyframe_render/first_frame_test/`。先看 `report.md` 和
`comparison-blind.jpg`；模型候选在 `review/`，输入、提示词、哈希与审图记录在
`_work/`。

### ControlNet 线稿诊断

为区分旧受控路线究竟受限于 `smooth_base` img2img，还是受限于 Canny 边缘，另有一组
纯 ControlNet 单变量对照：原稀疏 Canny 与详细合成语义线图使用完全相同的模型、
prompt、参数和四个 seed，且都不使用 img2img、strength 或 mask。

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \
  --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \
  --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.controlnet_line_test \
  --report
```

结果在 `output/keyframe_render/first_frame_test/controlnet_line_test/`。先看
`source-comparison.jpg`、`comparison-labeled.jpg` 和 `report.md`。

## Stage 1.2：输沙两关键帧

`stage1.2.md` 使用同一张自然化 sparse Canny 和相同 seed，生成“泥沙在河道内移动”
与“泥沙到达河口”两张一致关键帧。完整复现包括 raw 基线、一次有失败证据的提示词
修订，以及在 raw 模型无法定位悬沙时使用的机制软密度层：

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \
  --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \
  --generate --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \
  --generate-revision --force
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \
  --build-soft-sediment --base-seed 3102
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.transport_pair_test \
  --report
```

结果在 `output/keyframe_render/transport_pair/`。先看 `final/selected-pair.jpg` 和
`report.md`；`pairs-labeled.jpg`、`pairs-revised.jpg` 保留两轮 raw 失败证据。

## Stage 1.3：机制状态生成后续关键帧

`stage1.3.md` 把程序中的悬浮泥沙、水下沉积和新生陆地转成四张后续关键帧。
这套实现不是写死四张图的脚本：通用流水线负责规格校验、坐标投影、Canny、提示词、
候选管理、约束组合、验收和报告；`delta_causal` 适配器只负责解释当前三角洲程序状态。

```bash
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --prepare
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --generate
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --compose
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --evaluate
/opt/venv/bin/python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json --report
```

结果在 `output/keyframe_render/delta_sequence/`。先看 `report.html` 和
`sequence-contact-sheet.jpg`；报告逐帧展示程序状态、语义层、Canny 的四步来源、完整提示词、
16 张 SDXL/ControlNet 原始候选、最终组合、验收以及后续视频交接。

独立冒烟规格为 `keyframe_render/delta_sequence_smoke_spec.json`，它使用
display 40 / state 50 验证新增状态或不同帧数不需要修改通用流水线。

## Stage 1.2 补充：LTX-2.3 首尾帧过渡测试

`video_transition.py` 把 Stage 1.2 选中的 `in_channel.png` 和 `at_outlet.png`
分别接到 LTX-2.3 First-Last-Frame to Video 工作流的首帧与尾帧引导。当前首次测试为
512×320、24 fps、97 帧（约 4 秒），没有使用旧的纯文本视频冒充关键帧过渡。

先启动已经部署的 ComfyUI：

```bash
/persistent/ComfyUI/start-ltx2.3.sh
```

再从仓库根目录运行：

```bash
/workspace/comfyui-rocm-env/bin/python \
  -m modules.video_model.stage1.video_transition
```

结果在 `output/video_transition/ltx23_flf/`。先看 `transition.mp4` 和
`report.html`；`generated_frames.jpg` 展示 0 至 4 秒的 9 个采样时刻，
`workflow_api.json` 与 `metadata.json` 保存完整工作流和审计结果。

## 测试

```bash
.venv/bin/python -m pytest -q modules/video_model/stage1
```
