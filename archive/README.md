# 历史档案

当前实验从[项目首页](../README.md)进入。这里按用途保存已退出当前主链的文件。先按下表找到所属阶段，需要精确位置时，在[迁移清单](MOVED_PATHS.tsv)里搜索旧文件名或实验目录名。整目录移动的记录覆盖其下全部文件。

实验目录的内容、去重记录和复原方法见[实验结果存储说明](../docs/实验结果存储与复原.md)。

| 想找什么 | 去哪里 |
| --- | --- |
| 早期 FALCON 复现、12D 训练与旧 FPGA 对接 | [early_falcon](early_falcon/) 收录原根目录的采集、训练、评估、MPC 演示与交接说明。[experiments](early_falcon/experiments/) 按当时的实验名组织。 |
| 路线 A 的学习式决策与 Windows 调度实验、超网训练、早期路线 B 对照 | [历史研究源码](controller_architecture_study/) 沿用原文件名。对应的[研究文档索引](controller_architecture_study/docs/历史研究文档索引.md)可按实验现象、日期或模块查找。 |
| 以前给同事的交接、设计稿和进度 | [历史总看板](history/docs/协作进度.md)、[旧版首页](history/README_before_20261118.md)和[旧研究首页](history/README_study_before_20261118.md)。带日期的“最新”均指写作当时。 |
| 历史回归用例 | [tests](tests/) 中保留原测试名，可与同名模块、历史提交一起复现。 |
| 在这台机器运行过的旧数据、DNN 权重、原始日志 | 本地目录 `archive/local/` 保留目录结构，查询清单能得到具体位置。这里的材料随本机保存。Git 克隆只提供已纳入版本管理的内容。 |

## 要运行旧实验时

归档的代码保留了原导入名、命令行和相对于原工程的路径。阅读单个文件可以直接打开；复跑完整旧实验，优先在独立目录检出整理前的版本。在当前工程根目录运行下面的命令，其中目标目录应为一个新目录。

```powershell
git worktree add --detach ..\airsim_control_before_layout 3a513b4
```

然后按迁移清单将所需的**本地资产复制到该目录的原相对位置**，核对该实验的 manifest、模型契约及绝对路径绑定。原本仍留在当前实验区、但未随 Git 提交的资产也需从原目录单独复制。当前主链依赖的冻结模型交付、PS 迁移输入及近期故障证据保持原位置。

当前 Python 测试默认从 `tests/` 发现。保留的 FALCON 兼容性测试会从 `archive/early_falcon/experiments/` 读取历史配置，以便持续核对原协议。

历史可执行源码、模型、数据和归档清单分别保存。查到一个同名模型时，再用 release、输入维度、归一化与 SHA-256 确认身份。
