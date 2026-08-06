# EXP-S3-20260730-002

假设：合同声明缺失类型与空间关系后，Semantic Normalizer 可从未归属的 hard_boundary 分量恢复 bbox，再由通用 typed primitive 重建器出图；全程不读取 RGB 程序截图或外观参考。

证伪条件：若恢复对象数量错误、越界、控制图密度异常、跨学科器材失败，或 preserve_exact payload 改变任一字节，即失败。
