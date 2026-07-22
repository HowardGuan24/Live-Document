# 模块 B：解释型短视频生成

本模块接收文档模块产出的 JSON，生成 3–8 秒的 MP4、WebM、GIF，以及包含输入、完整提示词、参数、耗时、产物探测结果和失败信息的元数据 JSON。

## v1 工作流

```text
输入 JSON
  → 校验与 content_type 归一化
  → 解释型正/负提示词
  → Wan2.1-T2V-1.3B（AMD/ROCm + Diffusers）
      ↘ auto 模式不可用/失败时：procedural 可运行基线
  → ffmpeg 统一时长、帧率、尺寸和画幅
  → MP4 + WebM + GIF
  → ffprobe 完整性检查 + metadata JSON
```

生产候选采用 [Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)：模型卡给出的显存需求约为 8.19GB，官方 [Diffusers Wan 文档](https://huggingface.co/docs/diffusers/api/pipelines/wan) 提供原生 `WanPipeline`。在 48GB AMD GPU 上，使用与本机 ROCm 匹配的 PyTorch wheel 后余量充足。PyTorch 在 ROCm 上仍使用 `torch.cuda` 这一兼容 API，元数据会通过 `torch.version.hip` 记录实际 ROCm 版本。

`procedural` 不是视频模型，而是 CI、首次安装验证和模型故障时的解释动画基线。`--backend auto` 会优先使用 Wan；缺少模型依赖/GPU或推理异常时回退，并在元数据的 `fallback` 字段中明确记录原因。要验证真实模型且禁止回退，请使用 `--backend wan`。

## 安装

需要 Python 3.10+ 和 `ffmpeg`（包含 `ffprobe`、libx264、libvpx-vp9）。

仅运行本地基线：

```bash
cd /workspace/ai-concept-animator
python -m venv .venv
source .venv/bin/activate
pip install -r modules/video_model/requirements.txt
```

AMD/ROCm 模型后端：

1. 根据当前 ROCm 版本，在 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 选择 Linux / Pip / Python / ROCm，先安装匹配的 `torch`。不要用普通 PyPI 的 CPU wheel 覆盖它。
2. 安装模型层依赖并验证设备：

```bash
pip install -r modules/video_model/requirements-model.txt
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip)"
```

首次模型运行会从 Hugging Face 下载约 29GB 的仓库文件，需要网络和足够磁盘空间。若模型受本地 Hugging Face 配置限制，请先执行 `huggingface-cli login`。

## 输入

```json
{
  "source_text": "Gradient descent updates parameters along the negative gradient direction.",
  "visual_goal": "Help the user understand that a point moves step by step downhill on a loss curve.",
  "content_type": "process",
  "video_prompt": "A clean educational animation showing a point moving down a smooth loss curve."
}
```

三个文本字段必须是非空字符串。`content_type` 支持：

- `process`（也接受 `formula`、`operation`）
- `state_change`
- `data_flow`（也接受 `dataflow`）
- `scene`（也接受 `analogy`）

## 运行

无需下载模型的端到端冒烟测试：

```bash
python modules/video_model/generate_explainer_video.py \
  modules/video_model/samples/gradient_descent.json \
  --backend procedural \
  --run-name gradient-smoke
```

在 AMD GPU 上执行真实生成：

```bash
python modules/video_model/generate_explainer_video.py \
  modules/video_model/samples/gradient_descent.json \
  --backend wan \
  --seed 42 \
  --duration 5 \
  --run-name gradient-wan-s42
```

自动选择（适合演示环境）：

```bash
python modules/video_model/generate_explainer_video.py \
  modules/video_model/samples/data_flow.json \
  --backend auto
```

常用选项：

- `--width 832 --height 480 --fps 16`：默认输出规格；宽高必须能被 16 整除。
- `--seed 42`：固定随机种子。未指定时会生成种子，并写入元数据。
- `--cpu-offload`：显存不足时启用，代价是速度明显下降；48GB 卡通常不需要。
- `--loop-mode pingpong`：前半段后接倒放，得到严格往返循环。它可能破坏有方向的流程语义，默认关闭。
- `--failure-note object_drift`：记录人工观察到的问题，可重复传入。可选项见 `--help`。
- `--no-gif`：仅生成 MP4 和 WebM。
- `--overwrite`：允许覆盖同名运行产物。

成功产物位于：

```text
outputs/
├── video/<run-id>.mp4
├── webm/<run-id>.webm
├── gif/<run-id>.gif
└── meta/<run-id>.json
```

即使生成失败，`meta/<run-id>.json` 也会保留异常类型、消息、traceback、已用参数和总耗时。输出文件名默认包含 UTC 时间和种子，不会意外覆盖先前实验。

## 提示词策略

所有输入字段都会进入最终提示词。通用约束固定画面风格和摄像机，再按类型增加运动结构：

| 类型 | 正向结构 | 适合的动作 |
| --- | --- | --- |
| process | 单主体、单路径、按顺序完成 | 点沿曲线移动、一个步骤推进 |
| state_change | 同一对象的 before → after | 颜色、形状或状态渐变 |
| data_flow | source → transform → output | 少量粒子沿固定路径移动 |
| scene / analogy | 最多三个对象、一个关系 | 轨道、平衡、因果互动 |

负向提示词压制镜头移动、切镜、物体复制/形变、闪烁、杂乱背景和不可读文字。模型不可靠地生成标签，因此 v1 明确要求模型不要画文字；标签应在后续确定性覆盖层中加入。

## 测试

```bash
python -m unittest discover -s modules/video_model/tests -v
```

测试覆盖输入校验、四类提示词、自动后端检测，以及真实 ffmpeg 的端到端 MP4/WebM/GIF/metadata 生成。模型下载与视觉质量不属于自动测试，真实模型实验需按 `notes.md` 的检查表人工验收。

## 已知限制

- 概率视频可能漂亮但不能正确解释原文；文件完整不等于语义正确。
- Wan 文生视频对精确路径、对象恒常性和闭环没有保证。
- v1 没有自动语义评估或漂移检测；元数据列出必须人工检查的类别。
- 程序化回退只能提供类型级模板，不能替代模型结果，也不能证明模型环境安装成功。
- 模型权重和 ROCm/PyTorch 不写死在同一 requirements 文件中，因为 wheel 必须与主机 ROCm 版本匹配。

模型比较、适用内容和实验记录规范见 [notes.md](notes.md)。
