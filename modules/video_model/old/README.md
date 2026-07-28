# 模块 B V2：解释型短视频生成

模块 B 接收 `ExplanationSpec` 风格 JSON，生成 3–8 秒的原始视频、标准化 MP4、WebM、GIF 和可追溯 metadata。V2 的重点是让 W7900 + ROCm 上的速度/质量取舍可测量，而不是把文件可播放误当成解释正确。

## 后端选择

| 后端 | `auto` 顺序 | 优点 | 代价与适用边界 |
| --- | ---: | --- | --- |
| `ltx` | 1 | 2B distilled，少步推理；V2 主优化对象；支持可选首帧 I2V | 概率模型仍会漂移、复制物体或忽略因果关系 |
| `wan` | 2 | 保留 V1 基线，便于同案例横向比较 | 通常比 LTX distilled 慢；当前实现不支持 I2V |
| `procedural` | 3 | 无模型下载、稳定、适合 CI/安装检查 | 仅类型级模板，不理解具体语义，不是模型质量结果 |

`--backend auto` 按 LTX → Wan → procedural 选择；加载或生成失败都会继续尝试并记录原因。状态只有：

- `success_model`：显式选择的 LTX/Wan，或 `auto` 首选 LTX，实际成功且没有后端替换；
- `success_fallback`：`auto` 替换到 Wan/程序化后成功，或显式程序化产物成功；`actual_backend` 会继续区分两者；
- `failed`：没有得到有效产物。

若要验证某个模型且禁止回退，明确指定 `--backend ltx` 或 `--backend wan`。

## 工作流

```text
输入 JSON
  → 输入校验和 content_type 归一化
  → ltx-explainer-v2.1 正/负提示词
  → 可复用的 LTX / Wan runner（或程序化回退）
  → outputs/raw 原始模型 MP4
  → ffmpeg 尺寸、帧率、时长标准化
  → MP4 + WebM + 可选 GIF
  → ffprobe、SHA-256、完整性检查和 V2 metadata
```

LTX 使用 Lightricks 官方 Python pipeline 和 `ltxv-2b-0.9.8-distilled.safetensors`。W7900 路径使用 BF16 和 PyTorch 的 `cuda` 兼容 API，同时强制检测 `torch.version.hip`；不依赖 xFormers、FlashAttention、NVIDIA FP8 或自定义 CUDA 扩展。

## 安装

需要 Python 3.10+、ffmpeg/ffprobe，以及与主机 ROCm 匹配的 PyTorch。不要让普通 CPU/CUDA wheel 覆盖 ROCm wheel。

```bash
cd /workspace/ai-concept-animator
python -m venv .venv
source .venv/bin/activate
pip install -r modules/video_model/requirements.txt

# 按 https://pytorch.org/get-started/locally/ 安装匹配主机 ROCm 的 torch/torchvision
pip install -r modules/video_model/requirements-model.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
```

`requirements-model.txt` 固定官方 LTX 仓库 commit `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`，并使用同时能导入 LTX 与 `WanPipeline` 的依赖区间。首次运行需下载约 6.34 GB LTX checkpoint，以及官方 PixArt 仓库中约 19.05 GB 的 T5-XXL 文本编码器分片。若旧版 `huggingface-hub` 的可选 Xet 下载器在代理环境中失败，可临时使用：

```bash
HF_HUB_DISABLE_XET=1 python modules/video_model/generate_explainer_video.py \
  modules/video_model/benchmarks/speed_cases/speed_01_single_subject.json \
  --backend ltx --profile fast --no-gif --run-name ltx-smoke
```

`--vae-tiling` 对应官方 VAE 的空间 `enable_hw_tiling()`；`--vae-slicing` 对应按时间 latent 分片的 `enable_z_tiling(8)`。两者默认关闭，因为 48 GB 显存应先测不分片的速度基线。

## 输入与运行

```json
{
  "source_text": "Water evaporates, rises, cools, and condenses into clouds.",
  "visual_goal": "Show the upward transformation into a cloud.",
  "content_type": "process",
  "video_prompt": "Water vapor rises from one calm pool and gathers into one cloud."
}
```

`content_type` 支持 `process`、`state_change`、`data_flow`、`scene`，并保留 V1 别名。精确公式、符号和路径应优先交给确定性动画；`data_flow` 的 metadata 会明确提示这一风险。

真实 LTX 冒烟测试：

```bash
python modules/video_model/generate_explainer_video.py \
  modules/video_model/benchmarks/speed_cases/speed_01_single_subject.json \
  --backend ltx --profile fast --seed 1101 --no-gif --run-name ltx-fast-s1101
```

自动选择和多候选：

```bash
python modules/video_model/generate_explainer_video.py \
  modules/video_model/samples/gradient_descent.json \
  --backend auto --profile balanced --num-candidates 3 --batch-id gradient-batch
```

候选分别使用递增 seed 和 `gradient-batch-c01-s…` 文件名，共享 batch ID 与已加载模型；系统不会在没有评测方法时自动宣称“最佳候选”。

其它常用控制：`--failure-note object_drift` 可重复记录人工发现的问题；`--no-gif` 跳过 GIF；`--overwrite` 只覆盖同一安全 run ID 的精确产物路径。`--loop-mode pingpong` 会用倒放生成严格往返循环，但可能把有方向的因果/流程动作反转，因此默认仍为 `none`。

实验性首帧 I2V：

```bash
python modules/video_model/generate_explainer_video.py INPUT.json \
  --backend ltx --profile balanced --first-frame first-frame.png --run-name i2v-test
```

## 三个速度/质量 profile

| Profile | 分辨率 | 帧数/FPS | LTX 步数 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `fast` | 512×320 | 49 / 12 | 4 | 低成本预览 |
| `balanced` | 640×384 | 49 / 12 | 7 | 默认候选；完整 distilled 首轮 schedule |
| `quality` | 768×480 | 73 / 18 | 7+3 双阶段 | 较慢的 multiscale 最终候选 |

这些是初始可复现配置，不是预先假定的最终最优值。实测结果与接受/拒绝结论见 [V2 实验报告](benchmarks/V2_REPORT.md)。

Profile 的步数/guidance 是 LTX 参数。Wan 对照在未显式覆盖时保留 V1 的 30 steps / guidance 5.0，同时沿用相同案例、seed、提示词版本、分辨率、帧数和 FPS；metadata 会记录实际参数，避免把模型专用 schedule 假装成完全相同配置。

## 固定 Benchmark

速度集包含单主体运动、双物体交互、状态变化和自然场景；质量集包含 8 个自然过程/简单因果案例。runner 会在同一后端内复用模型，以区分首次 `model_load` 和后续 warm inference。

```bash
# 三 profile 的 LTX 固定速度基准
python -m modules.video_model.benchmarks.runner \
  --backend ltx --profile fast --profile balanced --profile quality \
  --case-set speed --no-gif

# 相同案例的 LTX/Wan 对照
python -m modules.video_model.benchmarks.runner \
  --backend ltx --backend wan --profile balanced \
  --case-set speed --no-gif

# 只跑固定案例，适合快速复现
python -m modules.video_model.benchmarks.runner \
  --backend ltx --profile fast --case-set speed \
  --case-id speed_01_single_subject --no-gif
```

输出位于 `benchmarks/results/`：每次运行生成 JSONL、聚合 JSON 和 Markdown 表。指标包括冷加载、warm inference、端到端时间、每生成一秒视频所需秒数、规格、步骤、峰值 VRAM、文件大小、后端、状态和模型 revision。

## 质量评测

[固定 rubric](evaluations/quality_rubric.json) 对语义忠实度、解释清晰度、物体一致性、镜头稳定、视觉简洁和循环适配分别给出 1–5 锚点，并记录漂移、复制、语义错误等失败标签。

AI 评测只是可选、离线兼容的结构化导入，不是 ground truth：

```bash
python -m modules.video_model.evaluation make-ai-packet \
  outputs/meta/RUN.json --frame frame-000.png --output outputs/evaluations/RUN-packet.json
python -m modules.video_model.evaluation validate-ai-review review.json
```

人工评测使用 [成对比较 CSV](evaluations/human_pairwise_template.csv)，比较 baseline/optimized、LTX/Wan、fast/balanced 或 T2V/I2V：

```bash
python -m modules.video_model.evaluation validate-human-csv \
  modules/video_model/evaluations/human_pairwise_template.csv
```

## Metadata 与产物

```text
outputs/
├── raw/<run-id>.mp4
├── video/<run-id>.mp4
├── webm/<run-id>.webm
├── gif/<run-id>.gif
├── meta/<run-id>.json
└── evaluations/
```

V2 metadata 保存 requested/actual backend、回退原因、模型与 revision、GPU/ROCm/PyTorch/库版本、最终提示词及版本、seed、profile、分辨率/帧数/FPS/步数/guidance、峰值显存、ffmpeg/ffprobe 命令、SHA-256、人工问题和评测引用。

计时字段固定为 `model_load`、`prompt_preparation`、`inference`、`vae_decode`、`postprocess`、`encoding_mp4/webm/gif`、`probe_validation` 和 `total`。官方 LTX/Wan pipeline 在一次调用中返回已解码帧，因此当前 `inference` 包含 VAE decode，`vae_decode` 为 `null` 并附说明，不伪造拆分数字。

## 测试

```bash
python -m unittest discover -s modules/video_model/tests -v
python -m py_compile modules/video_model/*.py modules/video_model/benchmarks/*.py
```

正常单元测试不下载模型；它覆盖 LTX/ROCm 检测、profile 和案例校验、计时 schema、显式回退、提示词版本、候选命名、结果聚合、人工模板与真实 ffmpeg 程序化端到端路径。

## 已知限制

- MP4 可播放、尺寸正确和哈希存在只能证明产物完整，不能证明解释语义正确。
- 概率模型不适合精确公式、算法轨迹或必须逐字正确的标签；文字应由后续确定性覆盖层添加。
- LTX/Wan 的 pipeline API 没有暴露独立 VAE decode 计时。
- `quality` multiscale 需要额外 upscaler checkpoint；速度和视觉收益必须分别测量。
- I2V 已接入官方 conditioning 路径，但保持 experimental，不能替代 T2V/Wan 的固定比较。
- 对外发布前需单独审阅 LTX checkpoint 的模型许可；代码依赖许可不等同于权重许可。
- 大型 raw/video/webm/gif 与临时评测产物默认不应提交 Git；只选择明确的小样本纳入版本控制。
