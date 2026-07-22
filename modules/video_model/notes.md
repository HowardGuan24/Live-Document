# v1 实验说明

## 候选方案

| 方案 | 优点 | 风险 / v1 结论 |
| --- | --- | --- |
| Wan2.1 T2V 1.3B + Diffusers | Apache-2.0；官方模型卡称约 8.19GB VRAM；原生 Diffusers；480p 短视频参数直接可控 | 仍有漂移和提示词偏离；选为 v1，48GB AMD 显存有充分余量 |
| CogVideoX-2B + Diffusers | 成熟的 T2V 接口；官方旧版 Diffusers 文档给出约 12.5–19GB 的优化后显存范围 | 固定的常用时长/分辨率约束较强，作为后续对照组 |
| LTX-Video I2V | 可用关键帧锁定构图；运动速度快，适合未来两阶段方案 | 模型版本和许可组合较多；还需实现并评估关键帧生成，v1 暂不引入 |
| ComfyUI | 节点生态丰富，便于快速试工作流 | 服务、节点版本和工作流 JSON 增加部署状态；首版选择直接 Diffusers CLI |

参考资料：

- [Wan2.1 1.3B 模型卡](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- [Diffusers Wan pipeline](https://huggingface.co/docs/diffusers/api/pipelines/wan)
- [Diffusers CogVideoX pipeline](https://huggingface.co/docs/diffusers/api/pipelines/cogvideox)
- [LTX-Video 模型卡](https://huggingface.co/Lightricks/LTX-Video)

## 为什么先用 T2V

更稳定的产品方向很可能是“可控关键帧 + I2V + 确定性标签层”。不过，首版首先需要验证概率视频路线本身是否对解释有效。Wan 1.3B 让我们用最少的模型集成代码得到基线，并通过同一元数据格式和固定种子开展对照实验。若漂移率或语义错误率不能接受，再把后端替换为 I2V，而不改输入、后处理和记录协议。

## 最适合概率生成的内容

- 场景/类比：允许视觉细节有变化，适合展示“围绕”“吸引”“扩散”等关系。
- 单一状态变化：主体少、变化连续、结果不依赖精确数值。
- 简单流程：只有一个对象和一条明显路径时可尝试。

不建议直接使用概率视频的内容：公式推导、带精确数值的数据图表、多分支数据流、代码执行、必须逐字正确的标签。这些更适合 Manim/SVG/CSS 等确定性渲染。

## 每次实验检查表

元数据先自动检查时长、尺寸和文件非空；人工应逐项检查并通过 `--failure-note` 记录：

1. `object_drift`：主体的位置、形状、颜色或数量非预期变化。
2. `incoherent_motion`：动作方向或因果关系不连贯。
3. `bad_loop`：首尾跳变明显。
4. `semantically_unclear`：画面好看但不能说明 `visual_goal`。
5. `prompt_ignored`：关键主体/动作缺失。
6. `noisy_or_cluttered`：背景或装饰抢占注意力。

建议同一输入固定其它参数，用 3–5 个 seed 比较。不要只保存“最好看”的一条；失败样本及其元数据是本模块的重要输出。

## 已观察结果

当前项目 `.venv` 没有安装 PyTorch/Diffusers，因此只完成了 procedural 后端的端到端验证。主机侧检查确认 GPU 为 gfx1100、47.98GiB VRAM，`/opt/venv` 中的 PyTorch 2.9.1 能通过 ROCm 7.2 识别设备；但该环境缺少 Diffusers，尚未下载约 29GB 的 Wan 仓库文件。现有验证证明输入、提示词、编码、转码、目录和元数据协议可以运行，不代表 Wan 的视觉质量或推理性能已经验证。首次模型实验后应在此追加每个 seed 的耗时和六类失败标记。
