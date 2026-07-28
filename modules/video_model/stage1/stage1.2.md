# Stage 1.2：输沙阶段的两张一致关键帧

本阶段沿用 Stage 1.1 已验证有潜力的纯 SDXL Canny ControlNet 路线，只生成三角洲
形成第一阶段的两张关键帧：

1. 泥沙随水流在河道内向下游移动；
2. 泥沙前缘到达河道出口，但尚未形成离岸羽流、沉积或新生陆地。

本阶段不制作视频，也不把模型纹理投回机制 mask。

## 1. 固定机制状态

从 `output/causal_delta/timeline.json` 按 display/state 双重匹配：

| 关键帧 | display | state | 悬浮颗粒 | 最前端 x | 海岸 x | 沉降 | 新陆地 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 河道内输送 | 10 | 21 | 308 | 29.9138 | 38 | 0 | 0 |
| 到达河口 | 13 | 27 | 392 | 37.8227 | 38 | 0 | 0 |

第一张的泥沙前缘仍距海岸约 8.1 个网格；第二张的最前端刚到海岸左侧。两张机制帧
只作为状态审计依据，不直接输入模型。

## 2. 强化后的 sparse Canny

两张图共用一张 1344×768 的稀疏控制图，只包含：

- 一条连续但轻微自然弯曲的海岸；
- 一条从左侧进入海岸的单河道；
- 两条轻微弯曲且在出口略微扩宽的河岸。

控制图不包含泥沙轮廓、羽流流线、水深线、等高线、颗粒、箭头、文字或 mask。
ControlNet 只锁硬几何，阶段差异由提示词表达。

## 3. 固定模型和参数

- SDXL Base 1.0 FP16；
- SDXL Canny ControlNet FP16；
- `StableDiffusionXLControlNetPipeline`，纯文生图 ControlNet；
- 1344×768，36 steps，CFG 6.5；
- ControlNet scale 0.60；
- seeds：3101、3102、3103、3104；
- 两张图使用相同控制图、相同 seed 和相同负向框架；
- 不使用 img2img、strength、mask projection 或 SDXL Refiner。

## 4. 实际提示词

### 关键帧 1：河道内输送

```text
photorealistic aerial view, strict orthographic top-down,
one river crosses sandy land from left into a blue-green coastal sea,
natural sand and water texture,
a dense ochre sediment current travels inside the river channel toward the coast,
sediment remains upstream of the mouth, clear seawater beyond,
no delta island, branches, text, or arrows
```

### 关键帧 2：到达河口

```text
photorealistic aerial view, strict orthographic top-down,
one river crosses sandy land from left into a blue-green coastal sea,
natural sand and water texture,
the dense ochre sediment current has reached the river outlet,
a compact turbidity front touches the mouth, clear seawater beyond,
no delta island, branches, text, or arrows
```

负向提示词只在阶段禁区上变化：

- 第一张禁止海中泥沙羽流和浑浊海面；
- 第二张禁止宽广或远离河口的离岸羽流。

两个 CLIP tokenizer 的 token 数必须在推理前检查且不超过 77。

### 有失败证据时的一次提示词修订

如果首轮 8 张保持了场景一致性，却都没有可见的赭色泥沙，则只修改泥沙措辞，不改变
控制图、模型、参数或 seed：

- 第一张明确写成锈褐色浑水从左侧填充到河道中段；
- 第二张明确写成锈褐色浑水从左侧填充到出口，前缘恰好终止于河口；
- 两张都保持下游海水为清澈蓝绿色。

该修订必须单独保留为 `sediment_emphasis`，不能覆盖首轮失败结果。

### 两轮 raw 结果都失败时的机制软层

如果首轮和提示词修订共 16 张 raw 结果都不能把褐色正确放进河水，则允许在最一致的
raw 图对上增加一个确定性软悬沙层，但必须满足：

- 密度直接来自两个选定 state 的全部粒子坐标；
- 两条 Canny 河岸之间的软区域只用于确认“这里在机制上是河水”；
- 先保留底图明暗，把河道内部统一为蓝绿色水体，再按粒子密度混合褐色；
- 不移动岸线，不生成新陆地，不把颜色层称为模型输出；
- 保存 density、alpha、河道水域约束、参数和哈希，并在报告中解释用途。

这不是旧阶段把任意模型纹理重新贴回所有语义 mask 的做法。它只解决 Canny 无法表达的
软悬沙状态，并且必须与 raw 候选分开存放。

## 5. 评审门槛

按相同 seed 成对评审，每对都回答：

1. 两张是否像同一地点、同一镜头和同一材质？
2. 第一张的赭色泥沙是否主要在河道内？
3. 第二张的泥沙前缘是否明显比第一张更靠近出口？
4. 第二张是否仍未提前出现宽广离岸羽流、沉积洲或分流？
5. 两张是否都保持自然画质，没有重新变成地图沟槽或塑料底图？

只有同时满足阶段可读性和场景一致性的同 seed 图对，才能进入后续插帧或视频阶段。

## 6. 输出

```text
output/keyframe_render/transport_pair/
├── report.md
├── report.html
├── source-comparison.jpg
├── pairs-labeled.jpg
├── comparison-blind.jpg
├── final/                 # 只有审图合格时才写入
├── review/
│   ├── in_channel/
│   └── at_outlet/
└── _work/
    ├── source/
    ├── prompts/
    ├── prepare_manifest.json
    ├── metadata.json
    ├── model_fingerprints.json
    ├── blind_order.json
    └── review.json
```
