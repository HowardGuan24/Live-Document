# Stage 1 Track A 报告：三角洲机制动画

状态：`PASSED`

## 1. 这份报告说明什么

这条流水线不是让视频模型猜测三角洲如何形成，而是先用一套可检查的网格规则计算
“输沙 → 减速 → 沉积 → 出水 → 分流”，再把计算状态画成动画。机制状态是事实来源；
渲染器只能读取状态，不能修改岸线、沉积厚度或水流。

本报告记录输入、算法、执行顺序、输出文件和验证结果。第一次接触项目的读者可从
第 8 节开始按命令复现。

最终媒体：

- [MP4 动画](delta_causal.mp4)
- [GIF 动画](delta_causal.gif)
- [八个代表帧](contact-sheet.jpg)
- [Track B 高质量关键帧报告](../keyframe_render/report.md)

## 2. 输入和代码入口

| 输入 | 用途 |
|---|---|
| [`config.py`](../../causal_delta/config.py) | 默认网格、随机种子、流速、沉降和水深参数 |
| [`primitives.py`](../../causal_delta/primitives.py) | 流场、粒子搬运、沉积、出水和绕流规则 |
| [`simulate.py`](../../causal_delta/simulate.py) | 运行 120 个机制状态并写入 JSONL |
| [`validate.py`](../../causal_delta/validate.py) | 在渲染前独立检查机制是否成立 |
| [`storyboard.json`](../../causal_delta/storyboard.json) | 把 120 个机制状态编排成 105 个展示帧 |
| [`render.py`](../../causal_delta/render.py) | 把状态画成 768×512 图片并叠加教学信息 |
| [`export.py`](../../causal_delta/export.py) | 编码 MP4/GIF、生成联系表、元数据和本报告 |

实际使用的完整参数保存在
[`mechanism/simulation_config.json`](mechanism/simulation_config.json)。关键参数如下：

| 参数 | 本次值 | 含义 |
|---|---:|---|
| 网格 | 96×64 | 机制计算分辨率 |
| 画布 | 768×512 | 输出图片分辨率 |
| 状态数 | 120 | 机制更新次数 |
| 随机种子 | 1909 | 控制粒子初始位置和横向扰动 |
| 每帧新粒子 | 14 | 泥沙输入量 |
| 上游流速 | 1.34 | 河道内的基准速度 |
| 海域最低流速 | 0.16 | 避免海域流场降为零 |
| 基础沉降率 | 0.02 | 减速后粒子的沉降概率基数 |
| 单粒沉积质量 | 0.12 | 每个沉降粒子增加的总厚度 |
| 沉积分散半径 | 2 | 沉积核半径，单位为网格 |

本次运行环境：Python `3.12.3`、NumPy
`2.1.2`、Pillow `11.0.0`、
imageio-ffmpeg `0.6.0`。

## 3. 一个机制状态是怎样算出来的

每个状态按固定顺序执行：

1. 根据当前陆地计算流场。河道内使用基准流速；入海后速度按离岸距离下降。
2. 注入 14 个悬浮泥沙粒子。
3. 粒子沿流场移动，并加入由 seed `1909` 控制的小幅横向扰动。
4. 粒子撞到新生陆地时，在附近下游水格中寻找替代路线。
5. 粒子越过河口保护区后，按局部减速程度计算沉降概率：

   ```text
   slowdown = 1 - local_speed / river_speed
   settling_probability = base_settling_rate × slowdown²
   ```

6. 沉降粒子通过归一化高斯核增加周围网格的沉积厚度；旧厚度不会减少。
7. 对每个水格应用唯一的出水规则：

   ```text
   new_land = not original_land and sediment_thickness > water_depth
   land = original_land or new_land
   ```

8. 新生陆地作为障碍反馈到流场，下一状态的水和粒子从其上下两侧绕行。
9. 把粒子、厚度、陆地、流场采样和统计量写成
   [`states.jsonl`](mechanism/states.jsonl) 中的一行。

这里的水深不是目标三角洲轮廓，而是连续的河口浅滩标量场。最终陆地仍只能由
“沉积厚度大于水深”这一条规则产生。

## 4. 为什么有 120 个状态，却只有 105 张展示帧

机制状态用于计算，展示帧用于讲解。`storyboard.json` 从每个阶段抽取部分状态，
再在阶段末尾停留 7 帧，给观众阅读时间。`timeline.json` 逐张记录展示帧对应的
机制状态，因此不会丢失追溯关系。

| 阶段 | 画面含义 | 状态范围 | 动态展示帧 | 末尾停留帧 |
|---|---|---:|---:|---:|
| `transport` | 河流输送悬浮泥沙 | 0–27 | 14 | 7 |
| `decelerate` | 河水入海后减速 | 28–49 | 12 | 7 |
| `accumulate` | 泥沙在水下逐层累积 | 50–100 | 25 | 7 |
| `threshold_change` | 沉积超过水深，沙洲出水 | 101–107 | 7 | 7 |
| `reroute` | 新生陆地使水流绕行分流 | 108–119 | 12 | 7 |

合计 105 张展示帧，12 fps，
时长 8.75 秒。

## 5. 渲染和视频编码

每张展示帧先从状态绘制海水、原有陆地、水下沉积和新生陆地，再叠加：

- 蓝色箭头：保存于状态中的流向和速度；
- 赭色圆点：悬浮粒子的固定抽样，避免粒子太密；
- 标题、图例、阶段字幕和进度点：仅用于讲解，不进入机制计算。

PNG 位于 [`frames/`](frames/)。MP4 使用 H.264、CRF 18、`yuv420p` 编码。
GIF 从整段动画抽样建立同一套 128 色全局调色板，并关闭抖动，以减少逐帧色闪。
实际 FFmpeg 命令保存在 [`metadata.json`](metadata.json) 的 `mp4_command` 字段。

## 6. 本次结果

- 120 个机制状态，经五段教学时间线编排为
  105 张展示帧。
- 正式画布 768×512，
  12 fps，时长 8.75 秒。
- 泥沙第 28 帧到岸，第 32 帧首次沉降。
- 第 101 帧首次出水；最终新生陆地 15 格。
- 最终 2 条通道，稳定尾段 19 帧。
- 模拟与渲染均未使用 GPU 或生成模型。

## 7. 自动验证

验证在渲染之前执行。任一门禁失败，`export.py` 会拒绝生成正式媒体。

| 检查 | 检查的含义 | 结果 | 本次证据 |
|---|---|---|---|
| `state_count` | 状态数量与配置一致，避免少算或多算帧。 | PASS | `{"actual": 120, "expected": 120}` |
| `thickness_monotonic` | 每个网格的沉积厚度只能增加，不能凭空消失。 | PASS | `"all cells non-decreasing"` |
| `new_land_monotonic` | 已经出水的陆地必须持续存在。 | PASS | `"all emerged cells persist"` |
| `land_threshold_exact` | 只有“沉积厚度大于当地水深”才能产生新陆地。 | PASS | `"land == original_land OR (thickness > depth)"` |
| `arrival_before_settling` | 泥沙必须先到达河口，之后才能发生沉降。 | PASS | `{"first_arrival": 28, "first_settling": 32}` |
| `visible_underwater_stage` | 出水前必须有足够长的水下沉积阶段。 | PASS | `{"first_underwater_deposit": 32, "first_emergence": 101, "lead_frames": 69}` |
| `emergence_in_threshold_stage` | 首次出水必须发生在教学时间线的出水阶段。 | PASS | `{"first_emergence": 101, "required_range": [101, 107]}` |
| `mouth_deceleration` | 河口平均流速必须明显低于上游。 | PASS | `{"upstream_mean_speed": 1.34, "mouth_mean_speed": 0.45545, "ratio": 0.33989}` |
| `final_channel_count` | 最终必须形成 2–3 条可辨认通道。 | PASS | `2` |
| `stable_channels` | 分流结果必须稳定至少配置要求的帧数。 | PASS | `{"tail_frames": 19, "required": 5}` |
| `new_land_connected` | 新生陆地必须是一个连通体，不是零散噪点。 | PASS | `{"components": 1, "cells": 15}` |
| `land_and_deposit_extent` | 陆地与水下沉积必须向海侧推进到最低距离。 | PASS | `{"land_front_x": 52, "deposit_front_x": 63}` |
| `state_traceability` | 每个状态都必须带帧号、阶段、流场采样和统计量。 | PASS | `"frame, beat, flow samples and stats present in every state"` |

完整机器可读结果见
[`mechanism/validation.json`](mechanism/validation.json)。

## 8. 从零复现

以下命令从仓库根目录 `Live-Document/` 执行。Track A 只需要 CPU：

```bash
python -m venv .venv
.venv/bin/python -m pip install -r modules/video_model/stage1/requirements.txt

.venv/bin/python -m modules.video_model.stage1.causal_delta.simulate
.venv/bin/python -m modules.video_model.stage1.causal_delta.validate
.venv/bin/python -m modules.video_model.stage1.causal_delta.export

.venv/bin/python -m pytest -q modules/video_model/stage1/causal_delta/tests
```

若系统没有中文字体，画面会自动使用英文字幕。要复现中文排版，请准备支持简体中文的
TrueType/OpenType 字体并在运行前设置：

```bash
export DELTA_FONT=/absolute/path/to/NotoSansCJKsc-Regular.otf
```

复现成功后应看到：

- `simulate` 输出 `state_count: 120`；
- `validate` 输出 `passed: true`；
- `export` 生成 105 张 PNG、MP4、GIF、联系表和报告；
- 测试全部通过。

也可运行：

```bash
.venv/bin/python -m modules.video_model.stage1.run
```

这个入口会重跑 Track A，并准备/评估 Track B 已存在的文件；它**不会自动执行耗时的
SDXL 候选生成**。Track B 的完整 GPU 命令见其
[报告](../keyframe_render/report.md)。

## 9. 文件如何追溯

```text
config.py
  ↓ simulate
mechanism/simulation_config.json + mechanism/states.jsonl
  ↓ validate
mechanism/validation.json
  ↓ storyboard + render
timeline.json + frames/*.png
  ↓ encode
delta_causal.mp4 + delta_causal.gif
```

渲染器只读取状态；岸线、沉积厚度、新生陆地、颗粒位置与流向均不能在渲染层反向修改。

| 输出 | 是否最终交付 | 用途 |
|---|---:|---|
| `delta_causal.mp4` | 是 | 正式 H.264 动画 |
| `delta_causal.gif` | 是 | 循环预览 |
| `contact-sheet.jpg` | 是 | 快速检查八个代表帧 |
| `frames/` | 否 | 视频编码源帧，可由状态重建 |
| `timeline.json` | 否 | 展示帧到机制状态的映射 |
| `mechanism/states.jsonl` | 否，但应保留审计 | 每个机制状态的事实记录 |
| `mechanism/validation.json` | 否，但应保留审计 | 自动门禁和证据 |
| `metadata.json` | 否，但应保留复现 | 编码参数、哈希、字体和耗时 |

## 10. 可复现性的边界

- 同一代码、参数、NumPy 版本和 seed 应产生相同机制状态。
- 字体文件、Pillow 或 FFmpeg 版本不同，可能使 PNG、MP4、GIF 的字节哈希不同，
  但不应改变机制状态和验证结果。
- 这是用于解释因果链的简化教学模型，不是经过观测数据校准的水动力或泥沙工程模型，
  不能用于真实工程预测。

本次 MP4：355047 bytes，
SHA-256 `e0ea7c9951dc598f88ac99871f9a91766f74353ab3718e734d680f1f2ae45e36`。

本次 GIF：2137539 bytes，
SHA-256 `cb21b3815dd20dcf40e0d17a0ca3b85bbaab0d630075e0201fb3eb3d1c6e3bc1`。

本次机制状态 `states.jsonl` 的 SHA-256：
`d77376df2cf0d556cfb52f7cc4053b375496f52d5e267ae2bb90e3760d0a3e6c`。
