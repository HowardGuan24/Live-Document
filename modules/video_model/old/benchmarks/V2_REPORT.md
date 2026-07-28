# Module B V2 实验报告

## 环境

- GPU：AMD Radeon PRO W7900 / Navi 31，47.98 GiB；
- ROCm：PyTorch 报告 `torch.version.hip = 7.2.53211-e1a6bc5663`；
- PyTorch：`2.9.1+gitff65f5b`；
- LTX：官方 `LTX-Video` commit `4b2d053057623ddd4d0a1d3e9cd28890e9ef487f`；
- checkpoint：`ltxv-2b-0.9.8-distilled.safetensors`；
- 精度：BF16；CPU offload 默认关闭。

核心权重加载检查：校验过 SHA-256 的 6.34 GB checkpoint 通过官方 loader 在 `9.363 s` 内把 transformer + VAE 以 BF16 放入 gfx1100，稳定分配 `5.932 GiB` VRAM。该检查没有文本编码器和去噪过程，因此只证明核心权重可在 ROCm 加载，不计为 `success_model`，也不用于宣称生成质量。

## 固定比较

结果下载/运行完成后由 `benchmarks/runner.py` 写入 `results/`。所有 LTX/Wan 横向比较必须使用相同 case、seed、profile 和提示词版本。

| 后端 | Profile | 成功 | 冷加载 | Warm inference | 峰值 VRAM | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LTX | fast | 待测 | — | — | — | 512×320，49 帧，4 步 |
| LTX | balanced | 待测 | — | — | — | 640×384，49 帧，7 步 |
| LTX | quality | 待测 | — | — | — | 768×480，73 帧，multiscale |
| Wan | balanced | 待测 | — | — | — | identical cases/profile |

### 基准管线校验（非模型结果）

2026-07-22 在固定 4 个 speed cases 上跑通 procedural 三 profile；结果文件为 `results/20260722T081840Z-speed.jsonl`。平均端到端时间为 fast `0.887 s`、balanced `1.025 s`、quality `1.639 s`。每组均为 `0 model success / 4 fallback success / 0 failed`，只证明 runner、转码、探测和聚合链路可用，不用于评价 LTX/Wan 速度或质量。

## 受控优化决定

| 实验 | 唯一主变量 | 速度证据 | 质量证据 | 决定 |
| --- | --- | --- | --- | --- |
| model reuse | 冷加载 vs 已加载 runner | 待测 | 固定 seed，预期像素不变 | pending |
| 4 vs 7 steps | inference steps | 待测 | rubric / pairwise 待填 | pending |
| 512×320 vs 640×384 | resolution | 待测 | rubric / pairwise 待填 | pending |

在速度数据和视觉复核完成前，不接受或拒绝任何生成质量优化，也不指定“当前最佳配置”。

## 下载故障记录

本机代理环境中的可选 `hf_xet` 对 checkpoint 分块请求返回 HTTP 416；`hf_transfer` 的并行 range 又返回 403。两次都属于权重传输失败，未报告为模型生成成功。标准 HTTP range 可下载，失败 metadata 保留在 `outputs/meta/`。这不改变生产安装默认；遇到相同问题可设置 `HF_HUB_DISABLE_XET=1` 或预先把验证过哈希的 checkpoint 放入 Hugging Face cache。
