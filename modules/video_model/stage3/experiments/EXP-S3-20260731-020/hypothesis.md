# EXP-S3-20260731-020

观察到的问题：BIO-01 的 Stage 2 正例看起来可用，但实现从程序截图按颜色寻找染色体，
没有读取结构化的染色体身份层；直接使用 SDXL 供体还会带入程序没有声明的细胞器。

证据文件：

- `visual_targets/BIO-01/manifest.json`
- `stage2/output/phase-7/route-b/BIO-01/all-variants-sequence.jpg`
- `stage2/phase7_hybrid_pbr.py`

失败类别：`state_renderer` 与 `appearance_condition`。

本轮唯一主要改动：保持供体、程序状态、对象绘制、颜色和所有门禁不变，只把
`region_material.transfer_mode` 从 `raw_underlay` 改为
`highpass_statistics`。

通用假设：对于由 `region + object_identity` 定义的有机对象，外观供体只贡献归一化的
高频材质统计，几何、拓扑和对象身份全部由程序语义层重建，可以减少外观向几何泄漏，
同时保留可见的材质增强。

证伪条件：

- 任一帧细胞连通数不是 `1,1,1,2`；
- 对象数不是 `6,6,12,12`；
- 任一时刻谱系单位总数不是 12；
- 最后两帧不是 6 个父对象各产生两条姐妹，且左右各 6；
- 程序允许区以外像素变化；
- 与正例/反例并排审查时，高频统计候选仍带入供体细胞器，或材质提升不可见；
- CHEM-01、MATH-02 重建哈希变化，或三角洲历史哈希失效。

固定 cohort：

- target：BIO-01；
- route regression A：CHEM-01；
- route regression B：MATH-02；
- historical regression：GEO-HIST-DELTA-01。

预算：

- 新图片模型候选：0（复用已冻结 SDXL 供体）；
- 确定性候选：2（一个负对照、一个候选）；
- 视频模型候选：图片通过后最多 2 个。
