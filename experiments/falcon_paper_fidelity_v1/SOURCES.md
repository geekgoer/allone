# 复现依据

## FALCON 论文

本地论文：`E:\BaiduNetdiskDownload\vitis-ai6.1\tmp\pdfs\FALCON.pdf`

- PDF 第 5 页（论文页码 410）：四旋翼动力学、规划器→MPC→DNN 数据流、`K=20`、12 维输入、4 维输出、10×40×5/0.05=40,000 条、80/10/10、六种网络、20,000 epochs、`1e-4`。
- PDF 第 10 页（论文页码 415）：AirSim 五门实验、epsilon 的 y-z 曼哈顿距离、50 ms 默认控制周期、ACADO/qpOASES。

## 作者引用和公开实现

- PAMPC 论文：<https://arxiv.org/abs/1804.04811>
- UZH RPG MPC：<https://github.com/uzh-rpg/rpg_mpc>
- Polynomial/Rapid Quadrocopter Trajectories：<https://github.com/markwmuller/RapidQuadrocopterTrajectories>

公开 `rpg_mpc` 默认配置给出了位置、姿态、速度、推力/角速度代价和输入边界；FALCON 论文没有附上其 AirSim 实验所用的完整 ACADO 生成代码。因此本目录的 CasADi 实现遵守公开动力学和这些公开默认值，但会在 manifest 中明确标记为端口实现，而不是作者二进制。
