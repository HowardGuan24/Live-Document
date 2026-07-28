# Stage 1.1：第一张图路线实验

本阶段只回答一个问题：三角洲动画的第一张高质量静态图应该怎么生成？

不制作视频，不测试首尾一致性。所有候选必须使用下面指定的本地模型，不能静默换模，
也不能把未经模型生成的程序图冒充模型结果。

## 1. 固定模型

- 文生图 / img2img：`stabilityai/stable-diffusion-xl-base-1.0`，FP16
- 结构控制：`diffusers/controlnet-canny-sdxl-1.0`，FP16
- 不使用 SDXL Refiner
- 不使用旧阶段的 mask projection；本实验要观察模型能达到的真实上限，不能把输出重新
  强行贴回像素语义区。

所有路线共用：

- 分辨率：`1344 × 768`
- 推理步数：`36`
- CFG guidance scale：`6.5`
- 调度器：模型目录中的默认调度器
- 输出格式：PNG

## 2. 固定机制状态

使用 `output/causal_delta/timeline.json` 的 display frame 25，对应 state frame 36。
这是“河水入海后减速”的早期状态：

- 注入颗粒：518
- 悬浮颗粒：516
- 已沉降颗粒：2
- 水下沉积网格：50
- 出水新陆网格：0
- 原始程序帧：`output/causal_delta/frames/0025.png`

选择理由：羽流已经明显进入海域，但仍没有新生陆地，符合“第一帧展示水流携沙、
沉积尚未形成三角洲”的叙事。

## 3. 要比较的四组

### Route 0：模型能力自检

目的不是生成三角洲，而是排除“本地模型或运行时本身坏了”。用一条常见、简单、
不依赖 ControlNet 的 SDXL 文生图提示词跑一个 seed：

```text
cinematic photograph of an astronaut exploring a dense jungle,
cool muted color palette, natural atmospheric light,
real fabric and vegetation texture, highly detailed
```

负向提示词：

```text
plastic toy, resin figurine, clay render, flat vector art,
pixel art, blurry, low detail, text, watermark
```

seed：`3100`。

### Route A：自由俯视生图

不输入程序底图，不使用 ControlNet。目标是测量 SDXL 在“俯视河口”提示词下的画质上限，
同时记录它是否会擅自改变河道、海岸线或沉积阶段。

正向提示词：

```text
photorealistic scientific aerial view, strict orthographic top-down,
one river enters a blue-green coastal sea from the left,
a broad ochre sediment plume spreads beyond the mouth through shallow water,
continuous coastline, realistic natural textures,
natural turbidity and depth gradients,
no emerged delta island, extra channels, text, arrows, boats, or buildings
```

负向提示词：

```text
pixel art, flat vector, infographic, schematic map, plastic, resin, clay,
glossy miniature, watercolor, blurry, dreamy, low detail, delta island,
extra islands, branching river, extra channels, oblique view, horizon,
perspective, boats, buildings, roads, people, text, labels, arrows, legend, watermark
```

seeds：`3101, 3102, 3103, 3104`。

#### Route C2：只在 Route C 失败时执行的镜头约束修订

如果 Route C 的 4 张都出现天空、地平线或海滩，说明模型把 river mouth / sandy
riverbed 理解成了水面以上场景。此时只修改“相机是否在水下”这一项，其他路线和参数
不动，补跑相同四个 seeds：

```text
underwater photograph, camera fully submerged, no sky or horizon,
blue-green water above a sandy bottom,
dense ochre silt current moving left to right,
volumetric sunbeams through suspended sediment,
realistic haze and backscatter,
turbid freshwater mixing with clear seawater, documentary, wide frame,
no fish, divers, coral, plants, text, or arrows
```

修订版负向提示词：

```text
above-water, aerial, top-down, sky, horizon, beach, shoreline, sandbar,
dry land, landscape, infographic, vector, pixel art, plastic, resin, clay,
abstract smoke, fantasy, fish, diver, coral, plants, text, arrows, blurry
```

这是一轮有失败证据的单变量迭代；如果 Route C 已经正确生成水下镜头，则不执行 C2。

### Route B：受控俯视生图

目的：在保留河道和海岸大结构的前提下，测试 SDXL img2img + Canny ControlNet 能否摆脱
像素风。

先从 state frame 36 生成三个输入：

1. `smooth_base.png`：连续、无文字、无箭头的陆地 / 河道 / 海水底图；
2. `plume_density.png`：根据 516 个悬浮颗粒坐标做高斯扩散得到的连续羽流密度图；
3. `coastline_canny.png`：只约束陆海岸线和河道边缘，不把颗粒或沉积斑点作为硬边缘。

`smooth_base.png` 会把羽流密度以半透明赭色混合进水体。它不是最终图，只负责给 img2img
提供连续色块和大致位置。Canny 只负责几何，不负责生成羽流纹理。

受控路线使用压缩后的完整提示词，确保也不超过 SDXL CLIP 的 77 token 上限：

```text
photorealistic scientific aerial view, strict orthographic top-down,
preserve the supplied single river mouth and continuous coastline,
a broad ochre sediment plume follows the supplied outflow into shallow blue-green water,
realistic earth and water textures, natural turbidity and depth gradients,
no emerged land, extra channels, text, arrows, boats, buildings, or people
```

实验采用单变量扫参，seed 固定为 `3102`：

- 固定 control scale `0.50`，测试 strength `0.55 / 0.70 / 0.85`
- 固定 strength `0.70`，测试 control scale `0.35 / 0.50 / 0.65`

其中 `strength 0.70 / control scale 0.50` 是共同基线，只生成一次。再用共同基线补跑
seeds `3101 / 3103`，检查结果是不是只对单一 seed 有效。共 7 张候选。

不测试 `strength 0.25 / 0.35`：旧结果已说明过低 strength 会让像素底图主导结果，
不适合本次“能否真正摆脱程序图观感”的问题。

### Route C：真实水下场景

不输入程序底图，不使用 ControlNet。它不再把机制帧渲染成地图，而是把同一阶段语义
翻译成“镜头在水里看见水流携沙”的场景，测试这是否比俯视地图更适合演示开场。

正向提示词：

```text
photorealistic underwater documentary at a river mouth,
camera just above the sandy riverbed,
a broad freshwater current carries dense clouds of fine ochre silt
into calm blue-green coastal water, physically plausible turbidity gradient,
soft volumetric sunlight, natural water, sand and silt textures,
wide cinematic frame, no animals, people, structures, text, or arrows
```

负向提示词：

```text
top-down map, infographic, vector art, pixel art, plastic, resin, clay,
miniature, abstract smoke, fantasy, fish, diver, coral, plants,
buildings, boats, text, labels, arrows, watermark, blurry, low detail
```

seeds：`3101, 3102, 3103, 3104`。

## 4. 输出目录

```text
output/keyframe_render/first_frame_test/
├── report.md
├── report.html
├── comparison-blind.jpg
├── comparison-labeled.jpg
├── review/
│   ├── model_sanity/
│   ├── free_topdown/
│   ├── controlled_topdown/
│   ├── underwater_scene/
│   └── underwater_revised/   # 仅 Route C 首轮失败时存在
└── _work/
    ├── source/
    │   ├── original_frame.png
    │   ├── smooth_base.png
    │   ├── coastline_canny.png
    │   └── plume_density.png
    ├── prompts/
    ├── metadata.json
    └── model_fingerprints.json
```

`review/` 只放需要人工查看的模型候选；`_work/` 放复现和审计需要、但不应当被误认为
最终结果的输入与元数据。每个文件的用途必须在报告中解释。

## 5. 评审规则

先看 `comparison-blind.jpg`，不根据路线名称预判。再看带参数的候选目录。

每条主路线至少回答：

1. 是否已经摆脱像素风？
2. 是否仍然塑料、发糊或像游戏地形？
3. 水流携沙语义是否一眼可懂？
4. 地貌 / 水体是否物理可信？
5. 能否直接作为后续关键帧视频的演示基底？

路线选择不是只看“最漂亮”：

- Route A 如果好看但结构漂移，必须明确写；
- Route B 如果结构稳定但仍然像贴图或塑料，必须明确写；
- Route C 如果最自然但失去俯视机制对应关系，必须明确写；
- Route 0 只用于判断模型健康，不参与主路线优胜比较。

## 6. report.md 必须包含

- 本次实际执行了什么；
- 为什么选择 state 36，以及上述状态数值来自哪里；
- 四组各自的输入、提示词、负向提示词、seed 和参数；
- 每个提示词在两个 SDXL CLIP tokenizer 下的 token 数，必须不超过 77；
- `smooth_base`、`plume_density`、`coastline_canny` 的生成方法和用途；
- 明确声明本实验没有使用 mask projection；
- 模型 ID、本地路径、FP16 权重哈希、软件版本、GPU、运行时间；
- 每张候选图与参数的对应表；
- 基于实际图片的优缺点和最终建议；
- 从仓库根目录开始可直接复制执行的复现命令；
- 全部输出文件的用途、哪些是最终查看项、哪些只是中间产物。
