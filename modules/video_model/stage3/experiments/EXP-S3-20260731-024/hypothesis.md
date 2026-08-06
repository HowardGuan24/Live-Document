# EXP-S3-20260731-024 — GEO-02 程序标量场回写

## 观察

GEO-02 已有的 terrain-only SDXL 候选能提供可信山体、阴天光照和空气透视，但它只是一张
静态外观图，不能证明迎风坡降雨、空气团移动或背风坡变干。把云和雨边缘一并送进
ControlNet 的历史候选又把它们误读成了山脊，属于控制编码失败。

## 本轮唯一假设

若冻结 terrain-only SDXL 图片作为外观底图，并把程序导出的
`geo02_humidity_cloud_rain` 标量场和 `geo02_parcel_identity` 身份层通过一个通用、
确定性的 state overlay 端口回写，那么可以保留自然山体外观，同时满足降雨位置、
空气团方向和雨影硬门。

负对照与候选只改变 `program_state_overlay_port_enabled`。底图、四个程序状态、相机、颜色、
阈值和输出尺寸在运行前冻结。

## 证伪条件

- 山体或相机随状态变化；
- 主降雨不在山顶左侧的迎风坡；
- 结果帧不是降雨峰值，或背风端降雨未明显下降；
- 空气团没有从左向右单调移动；
- 标量层把整幅图染白、形成规则网格或复制外观参考中的几何；
- GEO 通过但 PHYS-01、CHEM-01 的既有 State Renderer 哈希退化。

## Cohort 与预算

- target：GEO-02；
- route regression A：PHYS-01（共享连续场/高度数据）；
- route regression B：CHEM-01（共享标量场状态映射）；
- historical regression：GEO-HIST-DELTA-01；
- 新图片模型候选：0；
- 确定性图片候选：负对照 1 组、候选 1 组；
- 图片通过后的视频模型预算：L1 1 条，必要时 L2 1 条，随后确定性回退。
