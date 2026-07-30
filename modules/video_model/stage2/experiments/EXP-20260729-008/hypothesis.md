# EXP-20260729-008：为什么线稿 ControlNet demo 比我们的 Img2Img 更写实

EXP-007 的三条 Img2Img 路线都保住了器材，却几乎原样继承平涂初图。这个结果解释了
“有 ControlNet 为什么仍像程序图”的一半：ControlNet 只提供额外约束，Img2Img 初图
仍然决定模型能离开原像素多远。网上常见的“线稿逐步还原写实图”通常是 T2I
ControlNet：模型从噪声开始，只把线稿当作空间约束，不把平涂程序图当作外观起点。

本轮比较三组：

1. Img2Img + 稀疏 hard boundary：EXP-007 的四张基线，逐文件复用；
2. T2I + 稀疏 hard boundary：只给烧杯与滴定管白线，从噪声生成；
3. T2I + 关闭控制：同一 T2I 管线但 ControlNet 强度为 0，检验提示词本身能否稳定
   放对器材。

提示词、SDXL/ControlNet 权重、30 步、CFG 6.0、分辨率和四个复现编号相同。T2I 没有
Img2Img strength；为了元数据可比，配置仍保留原字段但管线明确记录为
`controlnet_t2i`。

假设：T2I + 稀疏 hard boundary 的玻璃和溶液材质显著优于 Img2Img 基线，并比 T2I
无控制更稳定地保持一只烧杯、一根滴定管和正视构图。

证伪条件：稀疏 T2I 少于 3/4 候选同时通过器材数量/构图门禁，或材质没有提高。

预算：12 张矩阵，其中 Img2Img 基线 4 张精确复用，只新生成 8 张；不运行视频模型。

