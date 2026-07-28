# Stage 1：三角洲机制动画与高质量关键帧恢复

## 1. 当前背景

服务器数据已丢失，现有 GitHub 仓库版本较旧。Stage 1 的目标不是恢复所有历史实验，而是先重建两项已经证明有价值的能力：

1. **机制驱动的三角洲解释动画**：用可验证的规则计算“输沙 → 减速 → 沉积 → 出水 → 分流”，再由 Python 渲染为 GIF/MP4。
2. **高质量关键帧生成**：从机制状态生成干净底图，再用本地图像模型提升视觉质量，同时用语义 mask 保持岸线、河道、沉积区和新生陆地不漂移。

Stage 1 暂不恢复完整 T2V/FLF2V 流程。历史实验已表明，直接 T2V 指令跟随差；三角洲这种缓慢形变中，视频扩散过渡的性价比也明显低于 RIFE。先把“正确机制”和“高质量静态关键帧”恢复好，再决定 Stage 2 是否加入视频过渡。

---

## 2. 建议目录

```text
Live-Document/modules/video_model/stage1/
├── stage1.md
├── causal_delta/
│   ├── config.py
│   ├── primitives.py
│   ├── simulate.py
│   ├── validate.py
│   ├── storyboard.json
│   ├── render.py
│   ├── export.py
│   └── tests/
├── keyframe_render/
│   ├── prepare.py
│   ├── enhance.py
│   ├── evaluate.py
│   ├── prompts/
│   └── tests/
└── output/
    ├── causal_delta/
    └── keyframe_render/
```

代码、状态数据、渲染和模型增强必须分层。**机制状态是事实来源，渲染器和生成模型不能反向修改机制。**

---

## 3. Track A：恢复机制驱动三角洲动画

### 3.1 要解释的因果链

```text
河流携带悬浮泥沙
→ 河水进入开阔水域后减速
→ 泥沙沉降并在水下累积
→ 沉积厚度超过水深后露出水面
→ 新生陆地改变流场
→ 水流绕行并形成分流
```

### 3.2 核心 primitive

- `flow_field`：根据当前陆地/障碍重新计算流向。
- `transport`：沿流场搬运悬浮泥沙颗粒。
- `decelerate`：河水进入开阔水域后降低速度。
- `accumulate`：沉降颗粒累积为连续沉积厚度。
- `threshold_change`：沉积厚度超过当地水深后变为新生陆地。
- `reroute`：新生陆地反馈到下一帧流场，水流绕行。

### 3.3 历史正式配置

先按以下参数恢复旧结果，再进行任何调参：

| 参数 | 值 |
|---|---:|
| 模拟网格 | 96 × 64 |
| 海岸线 | `grid x = 38` |
| 状态数 / 模拟帧率 | 120 / 12 fps |
| 随机种子 | 1909 |
| 每帧新增泥沙 | 14 |
| 河道速度 | 1.34 |
| 入海扩张率 | 0.35 |
| 海域最低速度 | 0.16 |
| 基础沉降率 | 0.02 |
| 单次沉积质量 | 0.12 |
| 沉积分散半径 | 2 |
| 横向微扰 | 0.055 |
| 河口保护列数 | 2 |
| 稳定通道门槛 | 5 帧 |
| 流场采样间隔 | 4 格 |

### 3.4 五段教学时间线

| 阶段 | 机制状态 | 展示帧 |
|---|---:|---:|
| 输送泥沙 `transport` | 0–27 | 14 动态 + 7 停留 |
| 河口减速 `decelerate` | 28–49 | 12 动态 + 7 停留 |
| 水下累积 `accumulate` | 50–100 | 25 动态 + 7 停留 |
| 沙洲出水 `threshold_change` | 101–107 | 7 动态 + 7 停留 |
| 绕行分流 `reroute` | 108–119 | 12 动态 + 7 停留 |

预期正式成片：**768 × 512、12 fps、105 张展示帧、约 8.75 秒**。

### 3.5 视觉编码

- 蓝色箭头：水流方向；箭头越短表示流速越低。
- 赭色颗粒：悬浮泥沙。
- 半透明褐色分层：水下沉积厚度。
- 暖绿色：露出水面的新生陆地。
- 阶段字幕、图例、进度点：只负责解释，不改变机制。

### 3.6 必须输出

```text
output/causal_delta/
├── mechanism/
│   ├── states.jsonl
│   ├── simulation_config.json
│   ├── simulation_summary.json
│   └── validation.json
├── frames/
├── delta_causal.mp4
├── delta_causal.gif
├── contact-sheet.jpg
├── timeline.json
├── metadata.json
├── report.md
└── report.html
```

GIF 使用单一全局调色板，优先 `max_colors=128`、`dither=none`，避免色彩闪烁。

### 3.7 自动门禁

至少验证：

- 沉积厚度单调不减；
- 新生陆地单调不减；
- 陆地严格满足 `original_land OR (thickness > depth)`；
- 泥沙先到岸，再发生沉降；
- 出水前存在可见的水下沉积阶段；
- 河口流速显著低于上游；
- 最终出现 2–3 条稳定通道；
- 新生陆地为连通体；
- 渲染结果、时间线和机制状态可逐帧追溯。

---

## 4. Track B：恢复并改进高质量关键帧生成

### 4.1 先恢复历史基线

选择同一地点的两个阶段：

- 首帧：`accumulate` 阶段结束，历史上为展示帧 71 / 状态 100。
- 尾帧：`threshold_change` 阶段结束，历史上为展示帧 85 / 状态 107。

必须通过 `timeline.json` 按语义查找，不能只硬编码帧号。

先从状态重新绘制 `clean_base`，去除：

- 箭头；
- 泥沙颗粒；
- 字幕；
- 图例；
- 进度点；
- 底部面板。

保留：

- 原始陆地；
- 海岸线；
- 河道；
- 海洋；
- 水下沉积厚度；
- 新生陆地 mask。

### 4.2 历史图像模型设置

恢复时优先使用：

- **SDXL Base 1.0 FP16**
- **SDXL Canny ControlNet FP16**
- 768 × 512
- 36 steps
- img2img strength `0.50`
- guidance scale `6.5`
- ControlNet scale `1.35`
- seed `3101`
- 先生成包含全部语义类别的尾帧，再用完全相同设置生成首帧。

旧环境中的模型路径曾为：

```text
/workspace/ai-concept-animator/.cache/models/sdxl-base-1.0
/workspace/ai-concept-animator/.cache/models/controlnet-canny-sdxl-1.0
```

新服务器先探测本地权重；缺失时明确列出，不要静默换模型。

### 4.3 几何硬约束

SDXL/ControlNet 的直接输出只作为“纹理提案”。正式增强图必须做 mask projection：

1. 将海水、原有陆地、水下沉积、新生陆地分成独立语义区域；
2. 提取模型在区域内部生成的纹理和明暗；
3. 将纹理投回原始语义 mask；
4. 在边界附近降低模型贡献，防止越界；
5. 岸线、河道、沉积区和新生陆地最终仍由机制状态决定。

### 4.4 当前未完成的问题

历史版本虽然摆脱了像素风，但视觉仍偏“模糊、塑料、低频贴图”。原因不是结构控制失败，而是目标风格过于含糊，且提示词中的 `soft watercolor / gouache / smooth` 容易产生柔化质感。

下一次不先生成视频，而是只拿**尾帧单图**测试三种明确风格：

1. **高质量物理地理教材插图**（主候选）  
   清晰地理边界、自然但克制的地貌纹理、浅海层次、可读的沉积羽流。
2. **博物馆科学信息图**  
   更精致的抽象形状，适合后续叠加箭头、标签和阶段信息。
3. **遥感/卫星图风格**  
   只作为写实上限对照，允许更丰富纹理，但不能牺牲解释性。

先每种风格生成 2 个 seed，只比较单张尾帧。选定风格后，才生成首尾帧对，并检查稳定区域风格差异。

推荐主提示词方向：

```text
high-end physical geography textbook illustration,
orthographic top-down coastal map,
scientifically accurate and visually clear,
crisp geographic boundaries,
subtle natural terrain texture,
layered shallow-water bathymetry,
soft suspended sediment plume,
restrained atlas color palette,
clean editorial cartography
```

避免：

```text
soft watercolor, dreamy, painterly, blurred edges, plastic surface
```

### 4.5 必须输出

```text
output/keyframe_render/
├── original/
├── clean_base/
├── style_search/
│   ├── textbook/
│   ├── museum/
│   └── satellite/
├── enhanced/
├── masks/
├── controls/
├── comparison.jpg
├── metrics.json
├── metadata.json
├── report.md
└── report.html
```

评估重点：

- 是否明显摆脱程序图/塑料感；
- 岸线、河道和语义 mask 是否保持；
- 首尾帧是否属于同一个场景和风格；
- 水下沉积与新生陆地是否仍能清楚区分；
- 是否适合重新叠加箭头、颗粒、字幕和图例。

---

## 5. 历史模型与恢复优先级

### Stage 1 必需

1. SDXL Base 1.0 FP16
2. SDXL Canny ControlNet FP16
3. Pillow / NumPy / SciPy（或等价数值工具）
4. FFmpeg / ffprobe / gifsicle

### 暂缓恢复

- **LTX-2.3**：旧报告记录的是 **22B distilled 1.1**，不是 23B；恢复前必须根据 checkpoint 文件名再次确认。它曾用于 FLF2V，对三角洲过渡耗时很高，Stage 1 不依赖它。
- **RIFE**：以后用于关键帧插值，Stage 1 静态图通过后再恢复。
- **Gemma 3 12B**：旧流程只把它作为 CPU 辅助 VLM；坐标评测失败，不参与机制和渲染，当前无需优先下载。
- **LTX-Video 0.9.8 2B / Wan2.1-T2V-1.3B**：历史 T2V 基线，连简单动作的语义跟随也不稳定，不作为当前主方案。

---

## 6. 建议执行顺序

1. 探测环境、模型权重和依赖，输出缺失清单。
2. 重建 `causal_delta`，先生成 `states.jsonl` 并通过机制门禁。
3. 恢复 V9 风格的程序动画、GIF/MP4 和报告。
4. 重建 `clean_base` 提取。
5. 恢复 SDXL + ControlNet + mask projection 历史基线。
6. 只对尾帧做三风格搜索。
7. 选定主风格后生成首尾帧对，并做一致性与几何评估。
8. Stage 1 验收后，再讨论 Stage 2 的 RIFE / FLF2V / V2V。

---

## 7. Stage 1 完成标准

Stage 1 只有同时满足以下条件才算恢复完成：

- 三角洲因果动画可以从零生成；
- 机制状态、渲染帧和最终媒体可追溯；
- 自动机制门禁通过；
- 正式 GIF 能清楚表达输沙、减速、沉积、出水和分流；
- 两张关键帧可以从机制状态重新生成；
- 图像模型只增强视觉，不改变科学几何；
- 至少找到一种明显优于旧“塑料感”结果的视觉风格；
- `report.md` 和 `report.html` 记录模型、参数、过程、结果和限制。
