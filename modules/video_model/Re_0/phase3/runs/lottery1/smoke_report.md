# Phase 3 最小测试报告 · lottery1

## 测试范围

- Phase 1 运动来源：`phase1/runs/lottery1` 的 brief、`cutoff_event`、20.3–26.3 秒过程、overlay 与字幕。
- Phase 2 视觉来源：`phase2/runs/lottery1` 的 `narrow_neck` 与 `cutoff_channel` 写实锚点。
- 仅生成一个相邻锚点片段：`narrow_neck → cutoff_channel`。
- 目标运动：洪水只在左中部侵蚀曲流颈，逐步建立短通道；旧大弯在片尾仍含水且连通，不提前形成牛轭湖。

## 实际运行

- 引擎：本地 ComfyUI LTX-2.3 First/Last Frame。
- 模型：`ltx-2.3-22b-dev-fp8.safetensors` + distilled 1.1 LoRA。
- 工作规格：512×288、24 fps、97 帧、4.0417 秒。
- 执行方式：单任务串行；动态显存、CPU VAE、6 GiB reserve 与 10 GiB headroom 沿用现有服务配置。
- 显存：运行期间服务上报的最低剩余显存约 15.9 GiB，没有 OOM。
- 本次采用单阶段 smoke 配置，没有运行高质量两阶段放大，以降低显存和时间风险。

## 两次结果

- 首次尝试保存在 `segments/01/attempt-1.mp4`，参数为 guide strength 0.7、image compression 25。失败点是曲流环内部出现会移动的深色湿痕，而颈部变化不够集中。
- 唯一定向重试把 guide strength 调到 0.85、image compression 调到 10，并明确锁死环内草地。重试清除了新增湿痕，镜头、岸线和植被总体稳定，选为最终 `segments/01/video.mp4`。
- 首次 prompt、生成元数据和预览均已保留；最终 prompt、API workflow、prompt ID、源图哈希和视频哈希也已保留。

## 输出判断

- `base_video.mp4`：无教学覆盖层的 LTX 写实基础片段，保留模型生成的 AAC 音轨。
- `final_video.mp4`：在基础片段上重新叠加 Phase 1 的起止 overlay，并烧录该时间段字幕。
- 有效部分：没有换场或镜头运动；大河道、旧曲流和周边地形身份保持一致；最终帧到达截弯后的锚点。
- 已知限制：512×288 下颈部开通变化仍较细微，河岸存在轻微生成式“呼吸”；这是验证链路的 smoke 结果，不是两阶段高质量成片。
- 后续建议：若继续下一轮，应先增加一个“颈部刚出现浅水缺口”的中间真实锚点，再对两小段分别生成；这比单纯提升分辨率更能强化拓扑变化。
