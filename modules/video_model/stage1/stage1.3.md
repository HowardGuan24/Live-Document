# Stage 1.3：基于程序机制图生成后续四张关键帧

本阶段接在 Stage 1.2 之后。已经选定的“泥沙刚到河口”关键帧保持不变，继续根据程序
机制状态生成后续四张图：

1. 泥沙进入海水后减速并形成紧凑的水下浑浊羽流；
2. 泥沙在河口外水下逐层累积，但仍未露出水面；
3. 沉积超过当地水深，同一浅滩的两个相邻湿沙斑块刚刚露出水面；
4. 两个出水斑块连接成沙洲，水流稳定地从它两侧绕行并分成两路。

最终得到五张连续关键帧：一张现有起点加四张新图。每对相邻图只承担一个主要机制
变化，供下一阶段继续使用 LTX-2.3 First-Last-Frame to Video 生成四段过渡。

本阶段只生成、审查和交付关键帧，不生成视频。

## 0. 双重目标：完成案例，同时形成通用程序

这次不能只为当前四张图写一段一次性脚本。交付物同时包含：

1. **案例结果**：三角洲从入海减速到稳定分流的四张新关键帧；
2. **通用框架**：把“程序机制状态 → 可解释控制 → 图像候选 → 机制约束 → 评估 →
   视频交接”做成由配置驱动的流水线。

通用不等于把当前三角洲的数字换成变量名。必须把三类内容真正分开：

- **框架固定逻辑**：读取关键帧规格、生成控制图、组装提示词、保存候选、合成语义层、
  计算评估、生成报告；
- **案例适配逻辑**：如何从当前 `states.jsonl` 读取 `particles`、`thick`、
  `new_land` 和 `flow_samples`；
- **案例配置数据**：state 编号、输出文件名、颜色、提示词差异、阶段禁区和验收规则。

框架代码中不得硬编码 state 49/100/107/119，也不得用文件名判断阶段语义。换一份规格
文件和状态适配器后，应能复用同一套流程制作其他程序动画的关键帧。

### 0.1 从现有实验中必须吸收的经验

通用方案要把已经验证过的事实固化成设计规则：

| 已有实验现象 | 不能继续采用的做法 | 框架中的通用改进 |
|---|---|---|
| 纯提示词无法稳定定位河道中的软泥沙 | 反复增加“muddy、sediment”等形容词 | 软物质位置来自程序状态，提示词只描述视觉材质 |
| 详细线稿会让结果变成硬沟槽或地图 | 把粒子、沉积纹理和所有轮廓都塞进 Canny | Canny 只保留岸线、物体边界等硬几何 |
| 每张图从噪声重新生成会改变海岸和镜头 | 把同一 seed 当作绝对一致性保证 | 固定视觉锚点，只重绘或合成真正变化的区域 |
| 不解释的 `mask` 和内部名称无法审计 | 只保存灰度文件和 JSON 参数 | 每个语义区域都保存图片、叠加图、普通话说明和来源 |
| 视频模型能遵守清楚的首尾帧，但不会替我们推理机制 | 让视频一次跨越扩散、沉积、出水和分流 | 把过程拆成相邻单变量关键帧，再逐段生成视频 |
| 把 Markdown 塞进 `<pre>` 的 HTML 没有可读性 | 只输出文字清单 | 原生可视化 HTML，按生成顺序展示真实中间图 |

上述规则不仅要写在报告中，还要落实到模块边界、文件结构和自动验证中。

## 1. 先明确程序图与成品图的关系

程序图负责回答“这一时刻发生了什么”，不是需要被模型照抄的画面。

程序图中有界面、标题、图例、箭头和离散粒子点。不要把完整程序截图直接输入模型，
否则这些调试元素可能进入成品。应从 `states.jsonl` 提取下列机制数据：

- `particles`：仍在水中的悬浮泥沙坐标；
- `thick`：每个网格已经累积的水下沉积厚度；
- `new_land`：沉积超过水深后真正露出水面的网格；
- `land`：原始陆地和新生陆地的总范围；
- `flow_samples`：水流方向和速度，用于判断水是否开始绕流；
- `stats`：悬浮、沉积、新生陆地、沉积前缘和通道数量的审计数字。

这些数据要先变成读得懂的中间图：

- **悬浮泥沙浓度图**：粒子越密集，图上越亮；
- **水下沉积厚度图**：只表示水底已经堆积的泥沙；
- **新生陆地图**：只表示已经露出水面的沙洲；
- **水流路径图**：只用于核对绕流与分流，不把箭头画入最终关键帧。

最终图可以使用这些中间图限制颜色和几何，但报告必须说明每张中间图的用途。不要只写
`mask`、`conditioning` 或内部文件名而不解释含义。

## 2. 固定视觉起点

现有起点不得重新生成或覆盖：

```text
output/keyframe_render/transport_pair/final/at_outlet.png
```

它对应 display 13 / state 27，含义是：

- 泥沙前缘刚到河口；
- 没有水下沉积；
- 没有新生陆地；
- 海水中还没有明显的离岸羽流。

后续四张图必须延续这张图的：

- 1344×768 分辨率和横向构图；
- 严格正交俯视镜头；
- 同一条河道、同一条海岸和同一个河口位置；
- 同一套沙地、水体、光照、色彩和纹理尺度；
- 河流从左向右进入海水的方向。

除河口外的动态水域、新生沙洲及其紧邻边界外，尽量直接保留起点图的像素。不要为每个
阶段重新随机生成整张地图；整图重生成会造成海岸、镜头和纹理漂移，使后续视频产生
“地面呼吸”。

## 3. 固定机制状态

从以下文件按 display/state 双重匹配：

```text
output/causal_delta/timeline.json
output/causal_delta/mechanism/states.jsonl
output/causal_delta/frames/
```

| 顺序 | 画面含义 | display | state | 悬浮泥沙 | 已沉积 | 水下沉积网格 | 新生陆地网格 | 沉积前缘 x | 通道 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 现有起点：刚到河口 | 13 | 27 | 392 | 0 | 0 | 0 | 无 | 1 |
| 1 | 入海减速扩散 | 32 | 49 | 678 | 22 | 145 | 0 | 50 | 1 |
| 2 | 水下沉积累积 | 64 | 100 | 1117 | 297 | 395 | 0 | 60 | 1 |
| 3 | 两个相邻沙斑刚出水 | 78 | 107 | 1155 | 357 | 434 | 9 | 61 | 2 |
| 4 | 绕流稳定分汊 | 97 | 119 | 1204 | 476 | 503 | 15 | 63 | 2 |

选择这些状态的原因：

- state 49 是“入海减速”节拍的末端，羽流已经清楚进入河口外，但尚无新陆地；
- state 100 是出水前最后一个状态，适合明确表现“水下有沉积、表面仍连续覆水”；
- state 107 已有 9 个新生陆地网格；原始网格是两个相邻连通分量，应表现同一浅滩的
  两个小出水斑块，不能为了文案方便强行提前连接；
- state 119 有 15 个新生陆地网格，两个斑块已经连接且双通道稳定，适合作为最终画面。

必须为五个状态保存程序图对比和机制数据摘要。程序图只用于审计，不直接作为成品。

## 4. 总体生成路线

### 4.1 固定背景，机制变化只发生在动态区域

以现有 `at_outlet.png` 作为视觉底图，不重新改变：

- 原始沙地；
- 固定海岸；
- 河道上游；
- 镜头、阴影和整体色调。

动态区域只包括：

- 河口外的悬浮泥沙羽流；
- 河口外的水下沉积；
- state 107/119 中由 `new_land` 指定的新生沙洲；
- 新生沙洲紧邻的两条绕流水道。

最终报告中要提供差异图，确认远离动态区域的地貌没有无故变化。

### 4.2 悬浮泥沙继续使用机制软浓度，而不是画离散圆点

把每个 state 的 `particles` 映射到现有自然化河口，再通过平滑扩散形成连续浓度：

- 近河口浓度较高，向海水方向逐渐变淡；
- 保留局部不均匀和柔和湍流感；
- 不把程序粒子直接画成橙色圆点；
- 不让褐色覆盖沙地；
- 不生成远离河口的大范围浑浊海面。

state 49 的羽流应紧凑、刚进入海水；state 100/107/119 可以更宽、更远，但仍要与程序
沉积前缘和粒子范围一致。

### 4.3 水下沉积必须看起来仍在水下

根据 `thick` 生成水下沉积层：

- 使用偏浅的沙褐色低对比纹理；
- 保留其上方蓝绿色水体、水面高光和透明感；
- 厚处比薄处更容易看见，但不能画出干燥沙地纹理；
- 水下沉积边缘应柔和，没有清晰陆地岸线和投影；
- state 49/100 的沉积都不能被误画成露出水面的岛。

必须在报告中并排展示：

1. 程序厚度图；
2. 映射后的水下沉积图；
3. 最终水下视觉；
4. 用一句普通话解释为什么它仍是水下沉积。

### 4.4 新生陆地必须由 `new_land` 决定

只有 state 107 和 state 119 可以出现露出水面的沙洲。

- 沙洲位置和轮廓来自 `new_land`，不能由提示词自由猜测；
- state 107 是同一浅滩上的两个相邻小型湿沙斑块；
- state 119 中两个斑块连接成一块沙洲并略有扩展，不能换位置；
- 新生陆地面积必须单调增加，不能在后一张缩小或消失；
- 表面使用湿沙、细颗粒和靠水边较深的自然纹理；
- 沙洲边界比水下沉积清楚，但不要出现硬塑料描边；
- 不生成许多离散岛屿，也不提前画成成熟三角洲。

可以在新生陆地边界上使用阶段专属的稀疏 Canny，也可以从现有沙地取样生成匹配纹理，
但无论使用哪条路线，都只能改变 `new_land` 及其少量羽化边界。必须保存模型原始候选、
最终组合图和区域说明，不能把组合结果冒充原始模型输出。

### 4.5 分流是水绕过沙洲，不是突然生成复杂河网

state 119 的两条通道必须对应程序中的稳定绕流：

- 主流在沙洲前分开；
- 水从沙洲两侧进入海水；
- 两路都保持下游方向；
- 可以用自然水面纹理暗示流向，但最终图不画箭头；
- 不生成树枝状分流网络；
- 不把原始上游河道劈成很多支流；
- 不增加第二块无依据的新陆地。

## 5. SDXL 与 ControlNet 的使用边界

继续使用已经部署并验证过的模型：

- SDXL Base 1.0 FP16；
- SDXL Canny ControlNet FP16。

ControlNet 的作用是约束需要保持的硬边界：

- 固定海岸和河岸；
- state 107/119 中真正出现的新生沙洲边界；
- state 119 中沙洲两侧的主要水路。

ControlNet 不负责决定悬浮浓度和水下沉积厚度。这两种软变化必须来自程序状态图。

可以先生成每个阶段的模型候选，用来寻找自然的水面、湿沙和水下质感；最终关键帧应以
现有起点图保持场景一致，并只把经过审计的动态区域合入。所有模型候选使用相同设置和
相同随机种子组。这里的“随机种子”只是让模型从可复现的噪声起点开始，不代表一种模型
或泥沙类型。

首轮建议保持 Stage 1.2 参数不变：

- 1344×768；
- 36 steps；
- CFG 6.5；
- ControlNet scale 0.60；
- seeds：3101、3102、3103、3104；
- 不使用 SDXL Refiner；
- 不静默换模型或降低精度。

如果需要局部纹理增强，必须明确记录：

- 哪张模型图提供了纹理；
- 哪个程序语义区域允许使用；
- 边界如何羽化；
- 哪些像素保持了原图；
- 最终图为什么被分类为模型原图或组合图。

### 5.1 Canny 控制图必须展示“怎么得到的”

报告不能只放最终 `canny.png`。每张 Canny 至少保留下列四步：

1. **机制几何源图**：从 `land | new_land`、河道和水路提取的干净黑白区域；
2. **自然画面投影图**：把 96×64 机制网格映射到 1344×768 视觉画面后的边界；
3. **最终二值 Canny**：实际输入 ControlNet 的黑底白线图；
4. **锚点叠加图**：把白线叠到现有 `at_outlet.png`，让人直接检查线是否画在正确位置。

Canny 的通用构建过程应为：

```text
程序状态中的布尔几何
→ 提取陆水交界和需要锁定的物体边界
→ 使用案例提供的坐标映射投影到成品画布
→ 去除短小噪声边和重复线
→ 生成单像素或轻微抗锯齿的稀疏边缘
→ 二值化为 ControlNet 输入
→ 计算边缘像素比例、连通段和叠加误差
```

具体规则：

- display 32 / state 49 和 display 64 / state 100 没有新生陆地，使用同一张固定河岸与
  海岸 Canny；
- display 78 / state 107 在固定 Canny 上增加由 `new_land` 得到的小沙洲边界；
- display 97 / state 119 使用同一沙洲的扩展边界，并核对两侧水路；
- 悬浮泥沙浓度、水下厚度纹理、流速箭头、标题和粒子点都不得进入 Canny；
- 不允许用一组手写像素坐标专门凑出本案例。坐标变换必须通过可配置
  `ProjectionSpec` 完成；
- 每张控制图保存尺寸、边缘像素数、边缘比例、SHA-256 和生成参数。

报告要用一句普通话解释：Canny 告诉模型“硬边界在哪里”，不告诉模型“泥沙有多浓”
或“水底堆了多厚”。

### 5.2 语言提示词必须展示“怎么组装的”

不要只展示最终的一长段英文。提示词程序要把每张提示词拆成：

1. **共用视觉锚点**：镜头、地点、光照、基础材质；
2. **本帧机制变化**：本状态新增或增强了什么；
3. **本帧禁止内容**：这一时刻绝对不能提前出现什么；
4. **共用质量负向**：界面、文字、塑料感、低质量和镜头漂移；
5. **最终组合文本**：真正送给两个 tokenizer 的内容。

每张图保存一个机器可读的 `prompt_parts.json`，至少包括：

```json
{
  "common_visual": "...",
  "mechanism_delta": "...",
  "stage_forbidden": "...",
  "common_negative": "...",
  "positive_combined": "...",
  "negative_combined": "...",
  "token_counts": {
    "tokenizer": {"positive": 0, "negative": 0, "limit": 77},
    "tokenizer_2": {"positive": 0, "negative": 0, "limit": 77}
  }
}
```

报告中先用中文解释每一部分的作用，再显示英文原文和 token 数。这样换一个机制案例时，
只需要替换“本帧机制变化”和“本帧禁止内容”，不必重写整个提示词系统。

## 6. 实际生图提示词

每张图由“共用提示词 + 阶段补充”组成。提示词只能描述视觉，不替代程序机制约束。

### 共用正向提示词

```text
photorealistic orthographic top-down satellite view,
same river mouth and sandy coast,
locked camera and shoreline,
natural sand, blue-green shallow water,
no text or diagram
```

### 关键帧 1：入海减速扩散，display 32 / state 49

```text
compact ochre turbidity plume just outside the river mouth,
concentrated near the outlet, softly fading into nearby sea,
subtle lateral spreading, no exposed sandbar or new land
```

### 关键帧 2：水下沉积累积，display 64 / state 100

```text
pale tan submerged deposit beneath continuous blue-green water at the mouth,
broader sediment plume, soft underwater edges, visible water surface,
no dry sand or exposed land
```

### 关键帧 3：沙洲刚出水，display 78 / state 107

```text
two adjacent wet sand patches just emerged at the river mouth,
parts of one shallow shoal, submerged deposit visible around them,
early emergence,
not a completed delta
```

### 关键帧 4：绕流稳定分汊，display 97 / state 119

```text
the same mouth sandbar persists and extends slightly,
river separates into two clear water paths around it,
sediment follows both downstream paths,
stable local bifurcation, no complex river network
```

### 共用负向提示词

```text
oblique view, horizon, camera change, shifted coastline, terrain morphing,
interface, legend, arrows, labels, point markers,
plastic CGI, vector edges, many islands, mature delta, complex river tree,
buildings, boats, people, watermark, blurry
```

阶段负向补充分别使用：

- state 49：`exposed sandbar, dry land shoreline, bifurcation`；
- state 100：`exposed sand, island shadow, dry shoreline`；
- state 107：`many islands, mature channels, large new land`；
- state 119：`three or more main channels, extra islands`。

推理前检查两个 CLIP tokenizer，正向和负向提示词都不得超过各自限制。若组合提示词过长，
应删除不影响阶段含义的风格形容词，不能截掉机制禁区。

以上文本按程序实际使用的逗号拼接方式，以当前本地 SDXL 的两个 tokenizer 实测：
四张组合正向提示词分别为 73、71、70、76 tokens；四张组合负向提示词分别为
67、64、65、64 tokens，均包含
special tokens 且不超过 77。

## 7. 候选、修订与失败处理

每张新关键帧先保留模型原始候选，再制作机制约束后的候选。不要覆盖失败结果。

评审顺序：

1. 先隐藏文件名和阶段做单图盲看；
2. 再按同一随机种子查看四张模型候选是否保持同一地点；
3. 再与现有 `at_outlet.png` 组成五帧序列；
4. 最后叠加程序数据检查机制位置。

只允许一次有证据的提示词修订。修订时一次只改一类问题，例如：

- 水下沉积被画成陆地；
- 沙洲太大；
- 分流变成复杂河网；
- 羽流跑得过远。

如果文字提示无法可靠定位悬浮泥沙、水下沉积或新生陆地，应优先使用程序数据生成的
确定性语义层，不要继续堆叠形容词。组合图必须与模型原始图分开保存并明确标注。

任何阶段未通过时，不得伪造“模型增强成功”的最终序列。报告应展示失败图、说明问题和
下一步，而不是只留下最好看的一张。

## 8. 关键帧验收门槛

### 8.1 场景一致性

五张图必须像同一个地点和同一台固定相机：

- 固定海岸、河岸和上游河道不漂移；
- 地面纹理和光照不闪变；
- 分辨率、色彩空间和裁切完全一致；
- 差异主要集中在河口外的水体、沉积、沙洲和绕流区域。

### 8.2 机制可读性

不看文件名时，第一次接触项目的人也应能按顺序看出：

1. 泥沙刚到河口；
2. 泥沙进入海水并减速扩散；
3. 水下沉积逐渐变厚但仍覆水；
4. 同一浅滩的两个相邻沙斑露出水面；
5. 两个沙斑连接，水流被形成的沙洲分成两路。

### 8.3 机制硬约束

- state 49 和 state 100 的新生陆地必须为 0；
- state 107 只能出现 9 个新生陆地网格对应的两个相邻出水斑块；
- state 119 对应 15 个新生陆地网格，两个斑块应连接且总面积不得缩小；
- 水下沉积范围从 state 49 到 state 119 单调扩展；
- 最终只有两条主要绕流通道；
- 没有提前出现成熟三角洲或复杂分流网络。

### 8.4 视频交接要求

五张最终图按固定顺序写入 `video_handoff.json`。每一对相邻关键帧都要说明：

- 视频中唯一应该发生的主要变化；
- 必须保持不动的地貌；
- 禁止模型在中间帧中生成的内容；
- 建议时长和视频提示词。

后续 LTX-2.3 过渡应拆成四段，不一次从“刚到河口”直接生成到“稳定分流”。

## 9. 通用关键帧流水线

### 9.1 规格文件

新增一个由配置驱动的 `sequence_spec.json`。它描述“要做哪些帧”，而不是在 Python 中
写死当前案例：

```json
{
  "sequence_id": "delta_formation_after_outlet",
  "canvas": {"width": 1344, "height": 768},
  "visual_anchor": "output/keyframe_render/transport_pair/final/at_outlet.png",
  "state_adapter": "delta_causal",
  "projection": "delta_naturalized_1344x768",
  "common_prompt": "...",
  "common_negative": "...",
  "keyframes": [
    {
      "id": "decelerated_plume",
      "display_frame": 32,
      "state_frame": 49,
      "mechanism_delta": "...",
      "stage_forbidden": "...",
      "semantic_layers": ["suspended_density", "underwater_deposit"],
      "geometry_layers": ["fixed_coast", "fixed_riverbanks"]
    }
  ]
}
```

四张新关键帧都按同一结构配置。JSON 中的省略项在实际文件里必须完整填写。

### 9.2 建议模块边界

在 `keyframe_render/sequence_pipeline/` 中实现可复用模块：

```text
sequence_pipeline/
├── schema.py            # 读取和验证 sequence_spec.json
├── adapters/
│   └── delta_causal.py  # 当前 states.jsonl 到通用语义状态
├── projection.py        # 机制坐标到成品画布的可配置映射
├── controls.py          # 几何源图、Canny、叠加图和验证
├── prompts.py           # 提示词分段组合与双 tokenizer 检查
├── candidates.py        # 调用 SDXL/ControlNet 并保存 raw 候选
├── semantic_layers.py   # 悬浮、厚度、新生陆地、流路等可视化层
├── composite.py         # 只在允许区域组合模型纹理和机制语义
├── evaluate.py          # 通用一致性检查与案例规则检查
├── report.py            # 从 manifest 生成可视化报告
└── cli.py               # prepare/generate/compose/evaluate/report
```

模块职责要求：

- `schema.py` 不理解三角洲，只验证通用字段；
- `delta_causal.py` 是唯一知道 `particles/thick/new_land/flow_samples` 具体含义的模块；
- `controls.py` 只处理硬几何，不读取提示词；
- `prompts.py` 只组装语言，不决定像素位置；
- `semantic_layers.py` 输出有名称、有单位或归一化说明的图层；
- `composite.py` 必须记录每个输出像素可能来自哪些输入；
- `report.py` 只消费前面保存的 manifest，重新生成报告时不得重新跑模型。

### 9.3 统一执行阶段

命令行至少支持：

```bash
python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json \
  --prepare
python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json \
  --generate
python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json \
  --compose
python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json \
  --evaluate
python -m modules.video_model.stage1.keyframe_render.sequence_pipeline.cli \
  --spec modules/video_model/stage1/keyframe_render/delta_sequence_spec.json \
  --report
```

`--prepare` 必须在不加载扩散模型的情况下生成程序审计图、语义层、Canny 全过程、提示词
组合和 token 检查。这样可以先发现坐标、控制图和语言错误，再付出模型推理成本。

每个阶段写独立 manifest，并记录输入哈希。重复运行时，输入未变化就复用已有结果；使用
`--force` 才重新生成。任何缓存命中都要在报告中可查，不能静默拿旧图当新结果。

### 9.4 通用评估与案例评估分开

通用评估包括：

- 文件尺寸、色彩模式和哈希；
- Canny 是否二值、稀疏且与画布一致；
- 两个 tokenizer 是否都未超限；
- 固定区域相对视觉锚点的差异；
- 模型 raw 图与组合图是否分开；
- 报告引用的每张图和 JSON 是否存在。

三角洲案例评估包括：

- 水下沉积和新生陆地是否按 state 单调增长；
- state 49/100 是否没有露出陆地；
- state 107/119 是否使用对应的 `new_land`；
- state 119 是否只有两条主要绕流通道。

不能把三角洲特有规则塞进通用评估模块。

### 9.5 至少验证一次可复用性

除正式四帧序列外，使用同一框架对一个非正式状态做 `--prepare` 冒烟测试，例如
display 40 / state 50。该测试不需要运行 SDXL，但必须证明：

- 规格文件可以新增一个状态而不修改通用代码；
- 能自动生成程序审计图、语义层、Canny 和提示词；
- 报告生成器能读取不同数量的关键帧；
- 输出不会覆盖正式序列。

这项测试的目的不是再挑一张成品，而是验证框架不是只对四个硬编码状态有效。

## 10. 输出目录

```text
output/keyframe_render/delta_sequence/
├── report.md
├── report.html
├── sequence-contact-sheet.jpg
├── source-comparison.jpg
├── video_handoff.json
├── final/
│   ├── 00_at_outlet.png
│   ├── 01_decelerated_plume.png
│   ├── 02_underwater_accumulation.png
│   ├── 03_sandbar_emergence.png
│   └── 04_rerouted_flow.png
├── review/
│   ├── raw/
│   └── mechanism_constrained/
└── _work/
    ├── source/
    │   ├── mechanism_frames/
    │   └── state_summaries/
    ├── semantic_layers/
    │   ├── suspended_density/
    │   ├── underwater_deposit/
    │   ├── new_land/
    │   └── flow_audit/
    ├── controls/
    │   ├── geometry_source/
    │   ├── projected_boundaries/
    │   ├── canny/
    │   ├── anchor_overlay/
    │   └── manifests/
    ├── prompts/
    │   ├── prompt_parts/
    │   └── combined/
    ├── manifests/
    │   ├── prepare.json
    │   ├── generate.json
    │   ├── compose.json
    │   └── evaluate.json
    ├── metadata.json
    ├── model_fingerprints.json
    ├── blind_order.json
    └── review.json
```

框架代码、`sequence_spec.json` 和案例适配器属于正式交付，不能只把生成图片放进
`output/`。

## 11. 报告必须展示完整生成过程

报告服务两个读者：

- 第一次接触项目、需要看懂“程序状态怎样变成一张图”的读者；
- 想把框架用于另一个程序动画、需要知道哪些部分可以替换的开发者。

`report.html` 不能只是把 Markdown 放进 `<pre>`。它至少应按以下顺序展示：

1. **最终结果**：页面顶部展示五张最终关键帧和一句话结论；
2. **整个流程图**：程序状态 → 语义层 → 几何控制 → 语言提示 → 模型 raw 候选 →
   机制约束组合 → 评估 → 视频交接；
3. **机制状态选择**：展示五张程序图、display/state、关键统计及选择原因；
4. **程序数据清洗**：展示从带 UI 的程序图到干净语义状态的过程；
5. **Canny 制作全过程**：机制几何源图、投影边界、最终 Canny、锚点叠加和边缘统计；
6. **语言提示词全过程**：共用视觉、本帧变化、本帧禁区、最终正负提示词及 token 数；
7. **模型实际看到了什么**：明确列出 ControlNet 输入和文本输入，并说明哪些中间图
   没有输入模型；
8. **模型原始候选**：展示同 seed 跨阶段对比、盲评图和失败证据；
9. **程序约束组合**：逐层展示悬浮浓度、水下厚度、新生陆地、允许修改区域和组合结果；
10. **前后差异与验收**：展示五帧差异热图、固定区域稳定性和机制硬约束；
11. **通用框架说明**：用一张表说明换案例时保留哪些模块、替换哪些 adapter/config；
12. **局限与失败**：诚实列出塑料感、模糊、边界、模型漂移和未解决问题；
13. **复现附录**：在折叠区域提供模型、参数、命令、哈希、JSON 和源文件链接。

每一张中间图都要回答“它是什么、为什么需要、是否输入模型、如何影响最终图”。不要留下
`seed 3102`、`mask`、`conditioning` 之类没有解释的内部术语。

### 11.1 报告中的逐帧流程卡

每张新关键帧都应有一张从左到右的流程卡：

```text
程序帧
→ 悬浮/厚度/新生陆地语义图
→ 本帧 Canny
→ 本帧正负提示词
→ 模型 raw 图
→ 允许修改区域
→ 最终关键帧
```

图片必须能点击查看原尺寸。技术参数可以折叠，但生成逻辑和关键中间图不能折叠到默认
不可见的位置。

### 11.2 报告中的通用化结论

报告最后不能只写“本次选了哪张”。必须回答：

1. 当前案例中哪些规则可以直接复用于其他程序动画？
2. 哪些逻辑属于三角洲 adapter，换案例必须替换？
3. 哪些参数目前仍是经验值，尚未验证通用性？
4. 如果程序没有粒子、厚度或新生陆地，应如何声明自己的语义层？
5. 如果案例没有适合 Canny 的硬边界，框架如何关闭 ControlNet 而不是伪造边缘？
6. 下一案例接入最少需要提供哪些文件和字段？

推荐给出最小接入契约：

```text
必需：
- 一个视觉锚点或明确的全图生成策略；
- 一组带唯一 ID 的机制状态；
- 每个状态的画布坐标或可配置投影；
- 至少一个可视化语义层；
- 每帧变化描述和阶段禁区。

可选：
- 硬几何边界（没有时关闭 Canny）；
- 连续密度或厚度；
- 新生对象区域；
- 流向或运动矢量；
- 单调性、连通性、数量等案例验收规则。
```

完成后运行 Stage 1 全部测试，并验证：

- 五张最终图均为 1344×768；
- 所有报告图片和链接存在；
- `video_handoff.json` 的顺序与文件一致；
- 每张图的输入状态、生成方式、参数和哈希都能追溯；
- Canny 的四步图和提示词分段记录完整；
- 通用模块中没有硬编码本案例 state 编号；
- display 40 / state 50 的 `--prepare` 冒烟测试通过；
- `--report` 可以独立运行且不会重新调用模型。
