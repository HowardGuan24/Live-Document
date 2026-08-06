# Phase 2 Fast Mode 报告 · lottery1

- Phase 1 来源：`phase1/runs/lottery1`，路线为 `realizable`。
- 生成引擎：内置图像生成工具；所有输出均由模型实际生成，没有伪造或复用其他历史 run。
- 选择锚点：`gentle_meander`、`expanded_meander`、`narrow_neck`、`cutoff_channel`、`oxbow_lake`，覆盖缓弯、曲流扩大、颈部变窄、洪水截弯和牛轭湖隔离五个必要状态。
- 世界参考：`expanded_meander`。该帧同时复用为自身的 `realistic.png`，用于统一晚春草地、深色河水、泥沙岸线、光照、垂直机位和真实感尺度。
- 最佳结果：`expanded_meander`、`narrow_neck` 和 `oxbow_lake`。前两帧的陆地颈连接状态明确；最终帧的弯月湖与活动河道清楚分离。
- 重试：`oxbow_lake` 首次结果在右下方的水陆间距不足，可能被误读为相连；使用唯一一次定向重试后已修正。首次结果保存在 `anchors/oxbow_lake/attempt-1.png`。
- 已知问题：逐帧植被纹理不能做到像素级一致；`cutoff_channel` 的洪水漫滩范围偏保守，但新通道、旧大弯和主河道的阶段语义可辨，足以作为后续视频化锚点。
