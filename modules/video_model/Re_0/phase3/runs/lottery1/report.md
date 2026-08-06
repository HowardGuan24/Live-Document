# Phase 3 完整阶段报告 · lottery1

## 来源与范围

- Phase 1 运动真相：`phase1/runs/lottery1` 的 brief、Bridge 时刻、`cutoff_event`、overlay 和字幕。
- Phase 2 视觉锚点：`gentle_meander`、`expanded_meander`、`narrow_neck`、`cutoff_channel`、`oxbow_lake`。
- Phase 1 与 Phase 2 全程只读，没有修改。
- 原单段 smoke 结果保存在 `smoke_base_video.mp4`、`smoke_final_video.mp4` 和 `smoke_report.md`。

## 完整时间线

1. `meander_growth`：缓弯扩大为成熟曲流，97 帧。
2. `neck_narrowing`：曲流颈继续变窄但保持陆地分隔，73 帧。
3. `01`：洪水冲开颈部并建立短通道，97 帧；使用 smoke 阶段的定向重试结果。
4. `partial_isolation`：旧河道两端开始淤积但尚未封死，73 帧。
5. `oxbow_completion`：两端完全封堵，弯月形水体与主河道分离，73 帧。

最终拼接时，后四段各裁掉一个重复首帧，共 409 帧、24 fps、17.0417 秒。

## 派生中间锚点

为了避免让 LTX 一次完成“泥沙增长 + 两端完全断开”，在 Phase 3 内新增 `partial_isolation`：

- 结构来源：Phase 1 的 30.5 秒 clean/overlay。
- 视觉来源：Phase 2 `world_reference.png`。
- 生成方式：内置图像生成工具，提示词保存在 `derived_anchors/partial_isolation/prompt.txt`。
- 语义：主捷径已经稳定，旧大弯仍有水，两端泥沙滩部分生长但保留水口。

该锚点只存在于 Phase 3，没有回写 Phase 2 manifest。

## LTX 运行

- 本地 ComfyUI LTX-2.3 First/Last Frame。
- `ltx-2.3-22b-dev-fp8.safetensors` + distilled 1.1 LoRA。
- 所有片段均为 512×288、24 fps；首尾引导 0.85、图像压缩 10。
- 单任务串行运行；最低上报剩余显存约 15.9 GiB，没有 OOM。
- 除截弯 smoke 片段外，其余四段首次生成即接受，没有消耗重试预算。

## 输出与判断

- `base_video.mp4`：五段去重后的写实基础视频，无教学覆盖层。
- `final_video.mp4`：逐段叠加 Phase 1 overlay 和压缩后的同步字幕。
- 为避免拼接点出现不连贯的生成音频，两个完整拼接视频均为纯视频；各段原始 `video.mp4` 仍保留 LTX 生成的 AAC 音轨。
- 最清楚的片段：`meander_growth` 与 `oxbow_completion`。
- 已知限制：`neck_narrowing` 和 `01` 的几何变化集中在很小区域，低分辨率下仍偏细微；部分岸线和草地存在轻微生成式“呼吸”。当前结果适合验证完整 Phase 3 链路，不是两阶段高清发布成片。
