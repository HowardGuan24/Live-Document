# Stage 1 Track B 报告：高质量关键帧生成

状态：`selected_pair_ready`

## 1. 这份报告说明什么

Track A 先用机制模型算出了三角洲变化，但程序渲染的画面纹理较简单。Track B 的任务是：

1. 从已经验证的机制状态中取出首尾两个关键帧；
2. 用 SDXL + Canny ControlNet 提议更自然的视觉纹理；
3. 用机制生成的语义区域图把模型限制回原来的海岸线、河道和沉积范围；
4. 比较六个尾帧候选，选定风格后再生成同风格首帧；
5. 输出可供下一阶段动画合成使用的首尾图片。

这意味着模型只负责视觉增强，不能决定“哪里是陆地”或“河道如何分流”。

最终结果：

- [最终首帧](final/first.png)
- [最终尾帧](final/last.png)
- [首尾对比图](final/comparison.jpg)

完整的机制动画生成过程见
[Track A 报告](../causal_delta/report.md)。

## 2. 全流程概览

```text
Track A 的 states.jsonl + timeline.json
  ↓ prepare.py：按教学阶段名称寻找首尾状态
base_images：去掉箭头和文字的机制底图
  ├─ control_edges：底图的 Canny 边缘，作为 ControlNet 输入
  └─ semantic_masks：五类机制区域，作为生成后的硬约束
  ↓ enhance.py：SDXL img2img + Canny ControlNet
raw_proposals：模型原始提案，只用于提供纹理，不是成品
  ↓ project_texture：按 semantic_masks 投回底图
constrained_candidates：几何仍由机制决定的候选
  ↓ 人工比较六个尾帧候选，选定 physical_geography_s3102
用同一 prompt、seed、参数生成首帧
  ↓ evaluate.py：几何门禁、首尾稳定性、联系表
final/first.png + final/last.png
```

对应代码：

| 文件 | 负责的步骤 |
|---|---|
| [`prepare.py`](../../keyframe_render/prepare.py) | 选状态、画底图、生成边缘图和语义区域图 |
| [`enhance.py`](../../keyframe_render/enhance.py) | 模型推理、纹理投影、候选与首尾帧生成 |
| [`evaluate.py`](../../keyframe_render/evaluate.py) | 指标、几何门禁、联系表、最终文件和报告 |
| [`prompts/`](../../keyframe_render/prompts/) | 三种正向风格提示词和公共负向提示词 |

## 3. 首尾关键帧从哪里来

输入不是动画 PNG 的截图，而是
[`states.jsonl`](../causal_delta/mechanism/states.jsonl) 中的机制状态。
`prepare.py` 读取 [`timeline.json`](../causal_delta/timeline.json)，按阶段名称取每个阶段
最后一个状态，避免以后时间线长度改变时选错帧：

| 名称 | 查找的教学阶段 | 展示帧 | 机制状态 | 选择原因 |
|---|---|---:|---:|---|
| 首帧 | `accumulate` | 71 | 100 | 水下沉积已形成，但新陆地尚未出水 |
| 尾帧 | `threshold_change` | 85 | 107 | 沙洲已经出水，适合表现阶段差异 |

每个状态重新绘制为 768×512 的 `base_images`。这一版底图关闭新陆地黄色网格边，
并去除流向箭头、悬沙粒子、标题、图例、字幕、进度点和底栏，只保留地貌语义。

## 4. control edges 和 semantic masks 到底是什么

### control edges

`control_edges` 是把底图转为灰度后，用 OpenCV Canny 阈值 100/200 得到的黑白边缘图。
它作为 `control_image` 送入 Canny ControlNet，告诉扩散模型主要边缘应当在哪里。
它不是最终效果图，也不会叠到最终图片上。

### semantic masks

`semantic_masks` 是从机制状态直接计算的黑白位置图，不是 AI 猜出来的标注：

| 区域 | 计算方式 | 为什么需要 |
|---|---|---|
| `original_land` | 模拟开始前的固定陆地 | 保持原始海岸和陆地范围 |
| `river` | 海岸线以内且不是陆地的水格 | 防止模型把河道填成陆地 |
| `ocean` | 不属于其他四类的水格 | 保持开阔水域 |
| `underwater_deposit` | 厚度 > 0.001、尚未出水且位于海岸外 | 保留水下沉积范围 |
| `new_land` | 机制状态中已经出水的格子 | 锁定新生陆地形状 |

五张 mask 必须互不重叠，并且合起来覆盖画面的每一个像素。程序会验证覆盖计数恰好为
1；有重叠或空洞就停止，而不是继续输出。

ControlNet 的边缘约束只能减少漂移，不能保证完全不漂移；semantic masks 是生成之后
执行的第二道硬约束。两者用途不同，所以都需要保留在复现材料中。

## 5. 模型、环境和精确参数

本次没有使用替代模型：

| 模型 ID | 本地路径 | 状态 |
|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | `/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0` | 可用 |
| `diffusers/controlnet-canny-sdxl-1.0` | `/workspace/ai-concept-animator/.cache/models/controlnet-canny-sdxl-1.0` | 可用 |

本次推理环境：

| 软件 | 版本 |
|---|---|
| `torch` | `2.9.1+gitff65f5b` |
| `diffusers` | `0.35.2` |
| `transformers` | `4.57.6` |
| `accelerate` | `1.13.0` |
| `safetensors` | `0.7.0` |
| `opencv-python-headless` | `4.13.0.92` |
| `GPU` | `AMD Radeon Graphics` |
| `HIP/CUDA runtime` | `7.2.53211-e1a6bc5663` |

模型参数：

| 参数 | 值 | 作用 |
|---|---:|---|
| 输出尺寸 | 768×512 | 与机制底图一致 |
| dtype | FP16 | 降低显存占用 |
| 推理步数 | 36 | 扩散去噪步数 |
| img2img strength | 0.50 | 保留底图结构同时允许生成纹理 |
| guidance scale | 6.5 | 提示词引导强度 |
| ControlNet scale | 1.35 | Canny 边缘约束强度 |
| 尾帧 seeds | 3101、3102 | 每种风格生成两个候选 |
| 选定 seed | 3102 | 首尾帧使用同一 seed |

选定的正向提示词来自
[`physical_geography.txt`](../../keyframe_render/prompts/physical_geography.txt)：

```text
high-end physical geography textbook illustration, orthographic top-down coastal map,
scientifically accurate and visually clear, crisp geographic boundaries,
subtle natural terrain texture, layered shallow-water bathymetry,
soft suspended sediment plume, restrained atlas color palette,
clean editorial cartography, fine print-quality detail, no labels, no arrows
```

公共负向提示词来自
[`negative.txt`](../../keyframe_render/prompts/negative.txt)：

```text
soft watercolor, dreamy, painterly, blurred edges, plastic surface, low-frequency texture,
pixel art, oblique perspective, aerial camera tilt, invented islands, shifted shoreline,
extra channels, text, labels, arrows, legend, frame, border
```

为识别本次实际使用的权重，五个主要 FP16 权重文件已记录 SHA-256：

| 模型 | 权重文件 | SHA-256 |
|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | `text_encoder/model.fp16.safetensors` | `660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `text_encoder_2/model.fp16.safetensors` | `ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `unet/diffusion_pytorch_model.fp16.safetensors` | `83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139` |
| `stabilityai/stable-diffusion-xl-base-1.0` | `vae/diffusion_pytorch_model.fp16.safetensors` | `bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68` |
| `diffusers/controlnet-canny-sdxl-1.0` | `diffusion_pytorch_model.fp16.safetensors` | `b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25` |

`--generate-candidates` 和 `--generate-pair` 会在实际推理前重新计算这些哈希；
若必需权重文件缺失，会直接停止。

机器可读版本见
[`_work/model_status.json`](_work/model_status.json) 和
[`_work/model_fingerprints.json`](_work/model_fingerprints.json)。

## 6. 六个模型候选是怎样生成的

先只生成尾帧，因为尾帧同时包含原始陆地、河道、海洋、水下沉积和新生陆地，
最适合暴露模型是否改变结构。三种风格分别是：

1. `physical_geography`：物理地理教材插图；
2. `museum_infographic`：博物馆科学信息图；
3. `remote_sensing`：遥感图风格。

每种风格使用 seed 3101、3102，共六张。模型调用实际收到四类主要输入：

```text
prompt          = 对应风格提示词
negative_prompt = 公共负向提示词
image           = _work/base_images/last.png
control_image   = _work/control_edges/last_canny.png
```

模型原始输出保存在 `_work/raw_proposals/`。这些图片可能改变颜色布局或地形，
所以明确标为 proposal，不直接进入 `final/`。

## 7. 模型纹理如何被限制回机制地形

`project_texture` 对每张原始提案执行以下处理：

1. 把模型提案缩放到机制底图尺寸。
2. 对提案做半径 7 的高斯模糊。
3. 用“原提案 − 模糊提案”取得中高频纹理残差。
4. 在每个 semantic mask 内计算提案的中位亮度，只允许 0.82–1.18 范围的局部明暗变化。
5. 每张 mask 向内腐蚀 4 像素，区分区域内部和边界。
6. 区域内部使用 0.62 的纹理权重；边界降到 0.14，减少跨岸线污染。
7. 逐 mask 合成；机制底图继续提供颜色类别和几何，模型的整体颜色布局被丢弃。

简化后的计算是：

```text
texture_residual = proposal - gaussian_blur(proposal, radius=7)
shading = clip(local_luminance / region_median_luminance, 0.82, 1.18)
textured = base × shading + 0.72 × texture_residual
final_pixel = base × (1 - weight) + textured × weight

weight = 0.62  # 区域内部
weight = 0.14  # 四像素边界带
```

所以最终图可以借用模型的细节和明暗，但岸线、新陆地和沉积区域仍来自机制 mask。
约束后候选保存在 `_work/constrained_candidates/`。

## 8. 为什么选 `physical_geography_s3102`

六张尾帧先排成：

- [模型原始候选联系表](review/raw-style-proposals.jpg)
- [约束后候选联系表](review/constrained-style-candidates.jpg)

选择不是由单一数值指标自动完成。本次人工比较的标准是：

- 海岸线和河道清楚；
- 不出现模型虚构的岛屿、文字或箭头；
- 陆地、水下沉积和新生陆地仍可区分；
- 纹理比程序底图自然，但不过分写实或抢夺教学信息；
- 适合后续重新叠加机制箭头、粒子和字幕。

按这些标准选定 `physical_geography_s3102`。之后用完全相同的模型、正负提示词、
seed `3102` 和推理参数分别生成尾帧与首帧，再执行同一套纹理投影。
原始首尾提案可在
[selected-pair-proposals.jpg](review/selected-pair-proposals.jpg) 对照。

## 9. 本次选择和检查结果

- 选定候选：`physical_geography_s3102`，即风格 `physical_geography`、seed `3102`。
- 生成模型：SDXL Base 1.0 FP16 + SDXL Canny ControlNet FP16。
- 尾帧比较了 6 个候选（三种提示风格 × 两个 seed）。
- 几何门禁：`通过`。
- 首尾帧未变化语义区域占比：`0.990723`。
- 这些未变化区域的平均 RGB 差异：`0.002159`；
  数值越低，表示首尾帧风格越一致。
- 最终首帧 SHA-256：`6d4d54e0fddc8ac3bdb89333455085958e40213588ada78a422f884df8eafdf7`。
- 最终尾帧 SHA-256：`e1254edac11e9402480cf23fd087c944bafb309a11c3a8fa48e1bb74aa6535b9`。

| 候选 | 平均梯度 | 对比度 |
|---|---:|---:|
| `museum_infographic_s3101` | 0.001582 | 0.048083 |
| `museum_infographic_s3102` | 0.001627 | 0.050883 |
| `physical_geography_s3101` | 0.001596 | 0.050538 |
| `physical_geography_s3102` | 0.001632 | 0.050420 |
| `remote_sensing_s3101` | 0.001564 | 0.048313 |
| `remote_sensing_s3102` | 0.001666 | 0.050902 |

`平均梯度` 和 `对比度` 只用于发现过度模糊或异常平坦的图片，并不自动决定哪张最好。
最终风格是根据约束后联系表人工选择的，判断重点是边界清晰、纹理克制、沉积与新陆地
仍能区分。选中风格后，才用相同 prompt、seed 和推理参数生成首帧。


几何门禁要求所有 semantic masks 互斥且完备、模型颜色布局未进入输出、边界模型权重
不超过 0.14。稳定性指标只比较首尾帧中语义类别没有变化的像素；它衡量风格一致性，
不代替人工检查或科学验证。

## 10. 从零复现

所有命令都从仓库根目录 `Live-Document/` 执行。

### 10.1 先复现 CPU 机制状态

```bash
python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt
.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export
```

必须先看到 `validation.json` 中的 `passed: true`，否则不要生成关键帧。

### 10.2 准备 GPU 环境和模型

先安装与你的 NVIDIA CUDA 或 AMD ROCm 环境匹配的 PyTorch；PyTorch 的安装命令取决于
显卡和驱动，不能跨机器统一。下面的 `/opt/venv/bin/python` 是本次服务器实际使用的
ROCm 环境；其他机器应替换成自己的 GPU Python。然后在同一个环境安装图像库版本：

```bash
MODEL_PYTHON=/opt/venv/bin/python
$MODEL_PYTHON -m pip install \
  diffusers==0.35.2 transformers==4.57.6 accelerate==1.13.0 \
  safetensors==0.7.0 opencv-python-headless==4.13.0.92
```

准备以下两个 Hugging Face 模型的 FP16 Diffusers 目录：

```text
stabilityai/stable-diffusion-xl-base-1.0
diffusers/controlnet-canny-sdxl-1.0
```

如果本机还没有权重，可用 Hugging Face CLI 下载完整仓库；代码会明确选择其中的
`variant="fp16"` 文件。完整快照占用空间较大，应先确认磁盘容量：

```bash
$MODEL_PYTHON -m pip install huggingface_hub
$MODEL_PYTHON -m huggingface_hub.cli.hf download \
  stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir /absolute/path/to/sdxl-base-1.0
$MODEL_PYTHON -m huggingface_hub.cli.hf download \
  diffusers/controlnet-canny-sdxl-1.0 \
  --local-dir /absolute/path/to/controlnet-canny-sdxl-1.0
```

默认路径是本报告第 5 节列出的路径。也可以放在其他位置并设置：

```bash
export SDXL_BASE_PATH=/absolute/path/to/sdxl-base-1.0
export SDXL_CANNY_CONTROLNET_PATH=/absolute/path/to/controlnet-canny-sdxl-1.0
```

检查环境；输出必须是 `status: ready`，并且 `substitution_used: false`：

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.prepare
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance --status-only
```

### 10.3 重新生成六个尾帧候选

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance \
  --generate-candidates --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
```

此时查看 `review/raw-style-proposals.jpg` 和
`review/constrained-style-candidates.jpg`。如果目标是复现本次选择，继续使用
`physical_geography` 和 seed `3102`；如果要重新选风格，应先记录选择理由。

### 10.4 生成选定风格的首尾帧并评估

```bash
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.enhance \
  --generate-pair --selected-style physical_geography --seed 3102 --force
$MODEL_PYTHON -m modules.video_model.stage1.keyframe_render.evaluate
$MODEL_PYTHON -m pytest -q modules/video_model/stage1
```

最后一条 `evaluate` 会把选中的约束后图片复制到 `final/first.png` 和
`final/last.png`，并重建联系表、指标和本报告。

注意：`python -m modules.video_model.stage1.run` 会重跑 Track A、准备关键帧并评估
已有模型结果，但为了避免意外占用 GPU，它不会自动执行
`--generate-candidates` 或 `--generate-pair`。

## 11. 输出文件和删除建议

| 目录或文件 | 实际用途 | 最终成品需要 | 能否删除 |
|---|---|---:|---|
| `final/first.png` | 动画合成使用的首帧，768×512 | 是 | 否 |
| `final/last.png` | 动画合成使用的尾帧，768×512 | 是 | 否 |
| `final/comparison.jpg` | 带标签的首尾快速对比，不是合成输入 | 复核时 | 可以 |
| `review/` | 原始提案和候选联系表 | 否 | 可以；不影响 final |
| `_work/base_images/` | SDXL img2img 的机制底图 | 重跑时 | 可以；`prepare` 会重建 |
| `_work/control_edges/` | ControlNet 的 Canny 输入 | 重跑时 | 可以；`prepare` 会重建 |
| `_work/semantic_masks/` | 几何约束和审计依据 | 重跑/审计时 | 可以；但会失去直接证据 |
| `_work/raw_proposals/` | 六张模型原始尾帧和生成记录 | 重新选风格时 | 可以；但需重新做模型推理 |
| `_work/constrained_candidates/` | 约束后候选及逐图哈希 | 重新选风格/审计时 | 可以；但需重新计算 |
| `_work/selected_pair/` | 选定风格的原始首尾模型图 | 审计时 | 可以；不影响现有 final |
| `_work/*.json` | 参数、模型状态、哈希和指标 | 复现/审计时 | 建议保留 |

复核图片：

- [模型原始尾帧候选](review/raw-style-proposals.jpg)：查看 SDXL 原本生成了什么。
- [受机制约束的尾帧候选](review/constrained-style-candidates.jpg)：比较可用风格。
- [选定风格的原始首尾提案](review/selected-pair-proposals.jpg)：检查约束前后的差异。

## 12. 可复现性的边界和已知限制

- seed 能固定本机推理，但更换 GPU 架构、PyTorch、Diffusers 或底层算子后，扩散结果
  可能不是逐字节一致；模型权重 SHA-256 用于确认最重要的输入一致。
- semantic masks 保证类别位置和边界来源于机制，但模型增强权重较克制，因此视觉提升
  主要是细纹理和局部明暗，不会变成完全写实的卫星照片。
- 候选选择包含人工视觉判断；平均梯度、对比度和 RGB 稳定性不是完整的美学评分。
- Track B 只生成两张静态关键帧，不生成二者之间的视频过渡。
- 本流程继承 Track A 的教学模型假设；它保证不篡改机制输出，但不把简化模型升级为
  真实水动力工程模拟。
    