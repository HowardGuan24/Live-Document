# Stage 1.2 输沙两关键帧生成报告

## 结论

最终采用“ControlNet 生成自然地貌底图，再由机制粒子决定泥沙位置”的组合方案。两张保持同一地点和镜头：第一张的褐色浑水前缘停在河口之前，第二张推进到出口；海域仍清澈，没有提前出现离岸羽流、沉积洲或分流。为了让两张底图尽量保持同一构图，使用了随机种子 3102；这个数字只是模型的可复现起点，不代表任何三角洲参数。两张图通过本阶段的叙事与一致性门槛，但不是纯模型直出；底图仍有平滑和规则感，适合进入运动设计验证，尚不应宣称是最终写实画质。

- 盲评：首轮先看 comparison-blind.jpg：8 张都有连续自然材质，但没有一张出现可辨认的赭色水中前缘；不同 seed 的地貌差异远大于同 seed 两阶段差异。修订版先看 comparison-revised-blind.jpg：仍没有可靠的褐色浑水，Candidate 04 的褐色位于陆地块而不是水体。解盲后，同 seed 图对在两轮中都高度相似，证明 shared sparse Canny + shared seed 对一致性有效，但 prompt 不能可靠定位软悬沙。
- 河道内阶段：最终第一张中，机制 state 21 的 308 个悬浮粒子形成连续软密度，褐色浑水从左侧沿单河道向右推进，前缘在河口前转回清澈蓝绿色；没有海中羽流和新生陆地。
- 到达河口阶段：最终第二张中，state 27 的 392 个悬浮粒子把褐色前缘推进到河道出口，河口外的大水体仍保持蓝绿色；没有宽广离岸羽流、沉积洲或分支。
- 一致性：两张使用 baseline seed 3102 各自对应的 raw 图、同一 Canny、同一水域约束和同一颜色算法。海岸、河岸、沙地纹理、色调和镜头高度一致，主要可读变化是悬沙前缘位置。
- 首轮 raw 结果：4 个 seed 的 raw 图对都保持了较好的同 seed 一致性，但 8/8 把河流理解成蓝绿色清水，阶段差异不可读。seed 3102 的左侧单河道与右侧开阔水体关系最清楚，因此只把它用作后续软层底图。
- 提示词修订结果：将抽象 sediment current 改成 rust-brown water fills river 后，8/8 仍未得到正确悬沙；seed 3102 的第二张把褐色生成在陆地上。修订失败，未选入 final。
- 软悬沙层结果：机制软层把 state 21/27 的全部粒子映射到自然化河道，生成两个不同前缘；显式河道水域约束修复 raw 图把部分河道内部画成沙色的问题，再进行保留明暗的蓝绿色水体归一化与褐色悬沙混合。这个步骤有实际用途且完整记录，不是未解释的 mask。

- seed baseline 3101：两阶段高度相似但海陆方向不清，河口语义弱，且没有赭色悬沙。
- seed baseline 3102：四对中地理关系最接近目标、阶段间最稳定，但两张都是蓝绿色清水。
- seed baseline 3103：画面主要是纵向水带与横向细线，目标单河口不清，且没有悬沙。
- seed baseline 3104：退化为纵向狭长水道，左侧入海关系丢失。
- seed sediment_emphasis 3101：措辞变化导致整体地貌变化，但仍没有褐色浑水前缘。
- seed sediment_emphasis 3102：第二张出现褐色陆地块而非水中悬沙，物理语义错误。
- seed sediment_emphasis 3103：两张仍是蓝绿色纵向水体，阶段不可读。
- seed sediment_emphasis 3104：两张仍是纵向水道，且没有目标泥沙变化。
- seed soft_sediment 3102：通过：同一河口、同一材质；前缘由河道内部推进到出口；没有提前进入海域。

最终采用“ControlNet 地貌底图 + 机制泥沙层”的组合方案。底图使用随机种子 3102；随机种子只是让模型从同一组随机噪声开始，便于两张图保持相似构图。

下一步：下一阶段可把这两张作为同场景端点测试短距离插帧或显式密度动画，但应让运动只推进悬沙密度，不要对整张地貌做生成式形变。若目标升级为最终写实画质，应先换成一张更自然且河道内部明确为水体的高质量参考构图，再沿用同一机制软层，而不是继续堆 prompt。

## 1. 实际做了什么

1. 从机制时间线选择 display 10 / state 21 和 display 13 / state 27。
2. 核对全部粒子坐标，保证第一张泥沙仍在河道上游，第二张刚到海岸左侧，两张都没有
   沉降和新生陆地。
3. 生成一张两帧共用的 `natural_sparse_canny.png`，只画自然海岸和两条河岸。
4. 使用纯 `StableDiffusionXLControlNetPipeline`，按相同 seed 生成首轮 8 张 raw 候选。
5. 首轮 8/8 没有赭色悬沙后，只修改泥沙措辞，再生成 `sediment_emphasis` 8 张 raw
   候选；模型、控制图、参数和 seed 不变。
6. 修订版仍失败后，选择地理关系最清楚的 baseline seed 3102，用 state 21/27 的真实
   粒子生成可审计的软悬沙密度层，得到当前两张混合关键帧。
7. raw 推理没有 img2img、strength、mask projection 或后期合成；最终混合关键帧额外
   使用了明确记录的河道水域约束和软颜色层，不能称为 raw 模型输出。
8. 两轮 raw 结果都先盲看单图，再解盲按 seed 成对审查。

## 2. 为什么选择这两个机制时刻

### 第一张：河道内输送

- display/state：10/21
- 悬浮颗粒：308
- 粒子前缘 x：29.9138
- 海岸 x：38
- 前缘距海岸：8.0862 个机制网格
- 到达或越过海岸的颗粒：0
- 水下沉积网格：0
- 新生陆地网格：0

### 第二张：到达河口

- display/state：13/27
- 悬浮颗粒：392
- 粒子前缘 x：37.8227
- 海岸 x：38
- 前缘距海岸：0.1773 个机制网格
- 到达或越过海岸的颗粒：0
- 水下沉积网格：0
- 新生陆地网格：0

`in_channel_mechanism_audit.png` 和 `at_outlet_mechanism_audit.png` 是对应程序帧，
只用于审计状态，没有输入模型。

## 3. sparse Canny 如何强化

共用控制图：`/workspace/Live-Document/modules/video_model/stage1/output/keyframe_render/transport_pair/_work/source/natural_sparse_canny.png`

- edge pixels：6227
- edge fraction：0.006033
- SHA-256：`f83271f98696bde63ef6546ae8493b8b4d289f27fb8240f9db89b94a7dbc6e4c`
- 包含：轻微自然弯曲的连续海岸、两条河岸、出口处小幅扩宽；
- 不包含：泥沙、羽流、水深线、等高线、颗粒、箭头、文字和 mask。

这次没有用详细线条控制泥沙。Canny 只回答“岸在哪里”，提示词回答“泥沙前缘走到
哪里”，避免把柔软的悬沙密度误生成硬沟槽。

## 4. 实际提示词

### 第一张正向

```text
photorealistic aerial view, strict orthographic top-down,
one river crosses sandy land from left into a blue-green coastal sea,
natural sand and water texture,
a dense ochre sediment current travels inside the river channel toward the coast,
sediment remains upstream of the mouth, clear seawater beyond,
no delta island, branches, text, or arrows
```

### 第一张负向

```text
oblique view, horizon, perspective, pixel art, flat vector, infographic,
schematic map, plastic, resin, clay, glossy miniature, watercolor, blurry,
low detail, delta island, branching river, extra channel, sediment plume in sea,
muddy ocean, beach, buildings, roads, boats, people, text, labels, arrows, watermark
```

### 第二张正向

```text
photorealistic aerial view, strict orthographic top-down,
one river crosses sandy land from left into a blue-green coastal sea,
natural sand and water texture,
the dense ochre sediment current has reached the river outlet,
a compact turbidity front touches the mouth, clear seawater beyond,
no delta island, branches, text, or arrows
```

### 第二张负向

```text
oblique view, horizon, perspective, pixel art, flat vector, infographic,
schematic map, plastic, resin, clay, glossy miniature, watercolor, blurry,
low detail, delta island, branching river, extra channel, broad offshore plume,
distant sediment plume, beach, buildings, roads, boats, people, text, labels, arrows, watermark
```

### 有失败证据后的 `sediment_emphasis`

第一张：

```text
photorealistic satellite image, orthographic top-down,
one river crosses sand from left into blue-green coastal sea,
natural water and earth texture,
opaque rust-brown sediment water fills river from left edge to mid-channel,
downstream river and sea stay clear blue-green,
no delta island, branches, text, arrows
```

第二张：

```text
photorealistic satellite image, orthographic top-down,
one river crosses sand from left into blue-green coastal sea,
natural water and earth texture,
opaque rust-brown sediment water fills river from left edge to outlet,
a rust-brown turbidity front ends at mouth, sea beyond stays clear blue-green,
no delta island, branches, text, arrows
```

共用负向：

```text
oblique view, horizon, perspective, pixel art, vector, infographic,
schematic map, plastic, resin, clay, miniature, watercolor, blurry, low detail,
dry riverbed, road, trench, canal, delta island, branching river, extra channel,
offshore plume, muddy sea, buildings, boats, people, text, labels, arrows, watermark
```

推理前 token 检查：

- `at_outlet.txt`：tokenizer 73/77，tokenizer_2 73/77
- `at_outlet_negative.txt`：tokenizer 75/77，tokenizer_2 75/77
- `at_outlet_v2.txt`：tokenizer 77/77，tokenizer_2 77/77
- `in_channel.txt`：tokenizer 74/77，tokenizer_2 74/77
- `in_channel_negative.txt`：tokenizer 75/77，tokenizer_2 75/77
- `in_channel_v2.txt`：tokenizer 69/77，tokenizer_2 69/77
- `sediment_emphasis_negative.txt`：tokenizer 76/77，tokenizer_2 76/77

## 5. 固定模型和参数

- SDXL：`stabilityai/stable-diffusion-xl-base-1.0`，FP16
- ControlNet：`diffusers/controlnet-canny-sdxl-1.0`，FP16
- pipeline：`StableDiffusionXLControlNetPipeline`
- size：1344×768
- steps：36
- CFG：6.5
- ControlNet scale：0.6
- seeds：[3101, 3102, 3103, 3104]
- scheduler：`EulerDiscreteScheduler`
- model load：5.333 秒
- 首轮 generation：137.749 秒
- 修订版 model load：5.355 秒
- 修订版 generation：137.696 秒
- GPU：`AMD Radeon Graphics`
- HIP：`7.2.53211-e1a6bc5663`
- raw 生成的 img2img / strength / mask projection：无 / 无 / 无

## 6. 全部候选

| 文件 | 版本 | 阶段 | display/state | seed | 秒 | SHA-256 |
|---|---|---|---|---:|---:|---|
| `in_channel_s3101.png` | baseline | in_channel | 10/21 | 3101 | 18.249 | `76ee5c3e9f18e27e16d196b28513a58cb2cf4d41cfd13984f289e57b1f8c80f0` |
| `at_outlet_s3101.png` | baseline | at_outlet | 13/27 | 3101 | 17.058 | `690f0a0338aebf4a25c224a5e3bd4e66d4beb97df6c8b4a3b5435c1cb858ae2e` |
| `in_channel_s3102.png` | baseline | in_channel | 10/21 | 3102 | 17.076 | `b3f347b38cf7c78984236fd004ee1ce2b14817abeb3a873e76a97dd458652458` |
| `at_outlet_s3102.png` | baseline | at_outlet | 13/27 | 3102 | 17.061 | `56527b4adda3028c1d769a45b89943bdda50cb1c1fb5b186bf895d6e21777fd9` |
| `in_channel_s3103.png` | baseline | in_channel | 10/21 | 3103 | 17.084 | `8966a387338db6e82b6a5263fee6c529d288ac9113201c193bc75829cdc9244b` |
| `at_outlet_s3103.png` | baseline | at_outlet | 13/27 | 3103 | 17.079 | `7409f95755a386ba1c433003e51c6e5be32f510cb5ff3743b4b443418f9cf041` |
| `in_channel_s3104.png` | baseline | in_channel | 10/21 | 3104 | 17.073 | `20b10c8d9a28a9e9a4689c03b86f85867bc9565e248a5b6cb1742d3f5a741b82` |
| `at_outlet_s3104.png` | baseline | at_outlet | 13/27 | 3104 | 17.066 | `35d0aac56b527b19a46aba6112ecd9cee3ae4cad83ccf59a7f4e0e6e06c6a9bc` |
| `in_channel_v2_s3101.png` | sediment_emphasis | in_channel | 10/21 | 3101 | 18.164 | `dbbe529275af9c2eb7fb4fe15cf7aaf68e8539f2cb029c6c3a0a28107a45cded` |
| `at_outlet_v2_s3101.png` | sediment_emphasis | at_outlet | 13/27 | 3101 | 17.008 | `7c33c602beec8e2c8f53a8b18894511f8252fa156fb0044a257ab6c97b4def02` |
| `in_channel_v2_s3102.png` | sediment_emphasis | in_channel | 10/21 | 3102 | 17.033 | `d1c2cadcbeb5f4bce2ca0f9ac4314eefceb14a51f9058269a64ddf6eb20c8a07` |
| `at_outlet_v2_s3102.png` | sediment_emphasis | at_outlet | 13/27 | 3102 | 17.055 | `1053dc9158cbeffae7a1efbae3696db9221db80a6cd34c9e44ccab697511fa35` |
| `in_channel_v2_s3103.png` | sediment_emphasis | in_channel | 10/21 | 3103 | 17.055 | `847f43b0fa7f60b3fcb6b1501675db0f72b24d843cc4902514e206a4c2390326` |
| `at_outlet_v2_s3103.png` | sediment_emphasis | at_outlet | 13/27 | 3103 | 17.050 | `981766f6e55d78275d58330635af0e308de8f19377e1d391536d2fc0f5d718d0` |
| `in_channel_v2_s3104.png` | sediment_emphasis | in_channel | 10/21 | 3104 | 17.113 | `a066b2ef281a907e3ee16e9a40dc69a2aa8ca7c8e01ade2226f75fc9edd81dc5` |
| `at_outlet_v2_s3104.png` | sediment_emphasis | at_outlet | 13/27 | 3104 | 17.207 | `5138587796d4a3d86c9a066c1280497ff4e95b393fe9a4c801ab55e4dfb139b3` |

查看顺序：

- `source-comparison.jpg`：两张机制审计帧和共用 Canny；
- `comparison-blind.jpg`：隐藏阶段与 seed 的单图盲评；
- `pairs-labeled.jpg`：每行一个 seed，左侧河道内、右侧到达河口；
- `comparison-revised-blind.jpg` / `pairs-revised.jpg`：提示词修订版；
- `pairs-soft-sediment.jpg`：机制软悬沙图对；
- `review/*/contact-sheet.jpg`：各阶段内部对比；
- `final/selected-pair.jpg`：只有图对通过门槛时才存在。

## 7. 为什么最终增加软悬沙层

选择 raw baseline seed 3102 作为同场景底图。对
state 21/27 的全部 308/
392 个粒子执行：

1. 将每个粒子的 x 从机制河道 `[0, coast=38]` 映射到自然化河道
   `[0, mouth=0.43×1344]`；y 相对机制河心 31.5、半宽 4 映射到当地弯曲河道。
2. 粒子落点后使用 OpenCV 高斯扩散 `sigmaX=20 / sigmaY=30` 像素，按正值第 99
   百分位归一化、`gamma=0.72`，再乘以边缘羽化 2.5 像素的河道区域。
3. 保留 raw 底图明暗，将河道水色以 alpha 0.62 向 `RGB(54,123,132)` 归一化，并在
   出口前 70 像素渐隐；随后按粒子密度以最大 alpha 0.72 混合
   `RGB(150,76,35)` 的悬沙色。

中间文件不是隐藏的：

- `river_corridor.png`：两条 Canny 河岸之间的软水域约束；
- `in_channel_density.png` / `at_outlet_density.png`：机制粒子密度；
- `in_channel_alpha.png` / `at_outlet_alpha.png`：最终悬沙透明度；
- `_work/soft_sediment_manifest.json`：全部输入、参数和 SHA-256。

这层不输入扩散模型，也不改变岸线。它先修正 raw 模型把部分河道内部画成沙色的问题，
再只在河道水域内改变颜色。最终结果必须称为“模型底图 + 机制软悬沙层”的混合图，不能
冒充原始 SDXL 输出。

## 8. 模型权重

- `stabilityai/stable-diffusion-xl-base-1.0`
  - `text_encoder/model.fp16.safetensors`：`660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd`
  - `text_encoder_2/model.fp16.safetensors`：`ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4`
  - `unet/diffusion_pytorch_model.fp16.safetensors`：`83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139`
  - `vae/diffusion_pytorch_model.fp16.safetensors`：`bcb60880a46b63dea58e9bc591abe15f8350bde47b405f9c38f4be70c6161e68`
- `diffusers/controlnet-canny-sdxl-1.0`
  - `diffusion_pytorch_model.fp16.safetensors`：`b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25`

## 9. 从仓库根目录复现

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

`_work/prepare_manifest.json` 保存状态选择、输入图与 prompt；`metadata.json` 保存每张
raw 候选的参数、耗时与哈希；`soft_sediment_manifest.json` 保存最终软层的完整方法；
`review.json` 保存人工审图结论。
