# 给正在阅读旧工程的Ubuntu Agent：保留成果，只纠正迁移边界

2026-11-16。请先阅读本页，然后以新包 `README_FIRST.md` 为唯一任务入口。

## 不需要从头重读

前面的交付粒度存在问题：大旧包带入了大量历史导入依赖，后来小包又只留下Python职责内核，没有把现有C++日历/采样/命令及板端驱动一起收齐。**这不是研究方案推倒，也不是要求把已经有的C++重写成Python。** 新统一包补齐现有源码、调用桥、构建、冻结模型、SDK和验收资料，核心算法没有因此改变。

旧目录、已写代码和笔记保留。新包解压到独立目录，不叠加覆盖旧文件。不要再花时间追整个旧工程的训练/CPU模拟链。原来的大旧包、小Python包、C++增量包都不再作为需要合并的运行来源。

## 你已经读过的东西，怎么复用

| 已读/已做的内容 | 现在怎么处理 |
| --- | --- |
| 三期限、版本、RAW237、保持界、义务调度核心 | 继续有效；新包 `source/ps_core` 与上次verified002的这部分逐字节一致 |
| `native_runtime/native_io.cpp` 的日历/采样/命令 | 继续复用；新位置 `platforms/windows/native_runtime/`，原C++字节不变 |
| `framework_native_io.py` | 原方法实现继续复用；新位置 `platforms/windows/`，仅两行导入换到新内核 |
| `pool237_runner.cc` 和header/DNNDK Pool | 继续有效；唯一副本在 `reference_assets/deployment/`；不重新实现量化与推理 |
| 旧Python训练/教师/CPU分段模拟、旧量化服务 | 停止追踪，不是PS运行目标 |
| 自己已经写的Linux C++移植 | 保留，按fixture核对后继续；不要因为换包就重写 |

具体旧文件/符号/哈希在 `MIGRATION_MAP.json`，不是要求再通读整个新包。可以比对旧解包根目录或其 `project`：

```bash
python3 -B tools/compare_prior_reading.py --old-root /path/to/old/extracted/package
```

`UNCHANGED_REUSE_READING` 表示已读文件的字节身份一致；`CHANGED_REVIEW_DIFF` 只需看差异，`NOT_IN_OLD` 表示旧快照没有该文件。脚本不修改旧目录。新生成模块的出处以 `source/SOURCE_MAP.json` 为准；类/函数已抽取并重排导入，不能把模块重排误认成全新算法。

## 本次必须新增阅读的三项

1. **执行责任表。** `README_FIRST.md` 第2节，明确释放、采样、输入、判断、派发、DPU、下发、完成事件在哪。新义务内核与原Windows原生IO尚未接通，Linux也尚未有完整编排；请完成迁移接线，不要假定已经提供现成PS主程序。
2. **命令符号。** Python客户端内部会翻转pitch/yaw，C++ AirLib不会。C++直接传物理 `wx,wy,wz`；按 `contracts/wire_command_cases.json` 核对最终RPC，不直接照抄Python层fixture。
3. **包内验收。** `python3 -B tools/check_unified.py`，只用新包的模型/黄金/计时/源码，不回读旧工程、不连接板子。通过后再做本地DPU和单机固定闭环。

## 下一步应实际做什么

按顺序完成：包内离线验收 → 复用真实Pool做本地DPU黄金 → 移植Windows IO的平台部分 → 接通新义务调度器与参考/RAW237/下发/完成事件 → 单机固定T50 → 双机固定/动态重复对照。

调度和输入构造的C++移植如已开始，继续原工作；模型、归一化、量程保护和冻结case不改。计划跳过仍默认关闭。完整PS程序不是只移植一个 `plan()` 函数；还包括到点释放、任务过期丢弃、状态—实际控制配对、真实DPU和命令回传。每个尚未接通的接口请记录，不用猜测填成“已实现”。

请先回复一张简短表：旧阅读哪些可复用、自己已写哪些保留、还缺哪些接口，然后继续实现。不要要求用户重新描述研究背景，不要删除旧进度，不要重新阅读几百MB工程。
