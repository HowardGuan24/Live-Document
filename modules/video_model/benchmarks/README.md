# Module B V2 benchmarks

这里的案例、seed、提示词版本和 profile 都固定，目标是让单变量速度/质量实验可复现。

## 目录

- `speed_cases/`：4 个速度案例；
- `quality_cases/`：8 个解释型质量案例，含必需元素、风险行为和预期 takeaway；
- `configs/`：`fast`、`balanced`、`quality` 及受控实验登记；
- `results/`：runner 生成的 JSONL、聚合 JSON 和 Markdown；
- `result_schema.json`：单条速度结果 schema；
- `V2_REPORT.md`：实测、接受/拒绝决定和局限。

## 运行

```bash
python -m modules.video_model.benchmarks.runner \
  --backend ltx --profile fast --profile balanced --profile quality \
  --case-set speed --no-gif

python -m modules.video_model.benchmarks.runner \
  --backend ltx --backend wan --profile balanced \
  --case-set speed --no-gif

python -m modules.video_model.benchmarks.experiment_runner \
  exp_steps_4_vs_7 --backend ltx
```

同一 `backend` 的 runner 会跨案例和 profile 保留模型。第一条成功记录的 `model_load` 是冷加载；后续记录应为 0，`inference` 是 warm 生成。LTX/Wan pipeline 的推理计时包含 VAE decode，metadata 会保留这一限制。

机器输出只回答速度、资源和文件完整性。视觉质量必须使用固定 rubric；AI 评分只能作为建议，接受质量优化至少需要留存可复核证据，最好使用人工成对比较。

## 实验纪律

每个实验只改一个主要变量，记录 baseline、变化变量、预期、速度、质量证据、结论和 `accepted`/`rejected`。不要同时改模型、seed、提示词、分辨率和步数后声称改进。
