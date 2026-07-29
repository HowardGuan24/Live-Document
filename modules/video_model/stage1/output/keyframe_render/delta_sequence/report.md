# Stage 1.3 机制关键帧生成报告

## 结论

五张关键帧使用同一张 Stage 1.2 视觉锚点。四张后续图的悬浮泥沙、水下沉积和新生陆地
来自对应程序状态，固定区域像素保持不变。当前阶段的 16
张 raw SDXL ControlNet 候选全部保留，但因全图漂移未直接用于最终像素。

生成链路是：

```text
states.jsonl
→ 悬浮浓度 / 水下厚度 / 新生陆地
→ 固定岸线加新陆地边界的二值 Canny
→ 分段正负提示词
→ SDXL + Canny ControlNet 原始候选
→ 固定视觉锚点上的机制约束组合
→ 通用检查 + 三角洲案例检查
→ LTX-2.3 首尾帧交接
```

## 最终序列

- `final/00_at_outlet.png`
- `final/01_decelerated_plume.png`
- `final/02_underwater_accumulation.png`
- `final/03_sandbar_emergence.png`
- `final/04_rerouted_flow.png`

## 模型与参数

- SDXL Base 1.0 FP16
- SDXL Canny ControlNet FP16
- 1344×768，36 steps，CFG 6.5，ControlNet scale 0.60
- seeds：3101、3102、3103、3104
- raw 生成耗时：274.430 秒
- 最近一次执行：缓存复用 16 张，重新生成 0 张

随机 seed 只是可复现的噪声起点。ControlNet scale 0.60 是边界约束权重，不是透明度。
每帧完整正负提示词和两个 tokenizer 的 token 数保存在 `_work/prompts/`。

## 模型原图为什么没有直接当成关键帧

16 张 raw 图证明稀疏 Canny 能让模型抓住河道、海岸和沙洲边界；但它不能锁定 Canny
之外的像素，模型把大片原有地貌重画成了浅色沙地。直接使用会造成镜头内地面“呼吸”。
因此 raw 图完整留在 `review/raw/` 和 `raw-candidates-by-seed.jpg`，最终图则保持同一视觉
锚点，只在 `allowed_region` 白色区域合入程序机制层。

## 复用框架

- 通用模块：规格验证、adapter 装载、投影接口、Canny、提示词编译、候选缓存、组合溯源、
  通用评估、HTML 报告和视频交接。
- 三角洲 adapter：解释 `particles`、`thick`、`new_land`、`flow_samples`。
- 案例配置：状态、投影、提示差异、颜色参数和验收规则。
- 没有硬边界的案例应关闭 ControlNet，不能把软浓度伪造成 Canny。
- 最小接入需要视觉锚点或全图策略、唯一状态 ID、坐标或投影、至少一个有说明的语义层，
  以及每帧变化与禁区。

## 验收

通用与案例检查共 11 项，结果：
`passed`。HTML 引用 76 个本地资源，缺失 0 个。
display 40 / state 50 的单关键帧 smoke 规格也可独立生成 `prepare-audit.html`。

完整可视化过程、Canny 制作、提示词拼装、语义层、候选图、验收和复现命令见
`report.html`。机器可读记录位于 `_work/manifests/`。
