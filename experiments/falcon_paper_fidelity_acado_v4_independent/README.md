# FALCON ACADO v4：独立状态查询数据集

这个隔离实验保留 FALCON 的 12 维输入、4 维控制输出、六种 FC/ReLU 网络和
ACADO/qpOASES teacher，但不再把一条五门飞行流程当作唯一的数据来源。

每一行独立采样：门相对位置、相对姿态、当前速度和当前加速度。随后使用固定
5 m/s 的局部五次多项式规划器生成 20 步参考，再由 ACADO 重新求一次控制标签。
加速度是多项式的初始边界条件，因此不再是无效占位。不可行状态会被拒绝后重采样，
不会复用其他状态的标签。

正式数据为 480,000 行，六类状态均衡。采集按 chunk 原子保存并支持 `--resume`。
拆分以 `query_group_id` 为单位做确定性 80/10/10 划分，组之间无泄漏。

训练和选模统一使用 normalized MSE；physical MSE 仍作为工程单位下的次要报告。
正式训练上限为 500 epochs，并允许在至少 100 epochs 后早停。旧实验使用的
20,000 epochs 只适用于当时更小的数据集，不应套用到 480,000 条独立样本。
先用 10x512 与 34x1024、各 3 个 seed 做容量试验。只有大模型平均 normalized
test MSE 至少改善约 10%，才建议训练完整六种架构。该设计提高出现容量趋势的概率，
但不会人为挑 checkpoint，也不能保证六个测试误差严格单调。

此数据生成和 DNN 训练都不需要启动 AirSim。

## 推荐执行顺序

以下命令均从项目根目录执行。正式采集和训练命令可重复执行：采集器跳过已经原子
完成的 chunk；训练器跳过已有 `metrics.json` 的完整模型，并从其余模型的
`latest_checkpoint.pt` 恢复。

```powershell
$proj = 'F:\FalconReproduction\airsim_control'
$trainPy = 'D:\files\PYTHON\cv_lab\Yolov8_Beginner_Demo-main\.venv\Scripts\python.exe'
Set-Location -LiteralPath $proj

& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\generate_state_queries.py --config .\experiments\falcon_paper_fidelity_acado_v4_independent\state_query_config.json --resume

& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\prepare_state_query_dataset.py --teacher-root .\experiments\falcon_paper_fidelity_acado_v4_independent\teacher_state_queries_480k --output-dir .\experiments\falcon_paper_fidelity_acado_v4_independent\dnn_dataset --seed 32025

& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\train_capacity_models.py --mode pilot --data-dir .\experiments\falcon_paper_fidelity_acado_v4_independent\dnn_dataset --output-root .\experiments\falcon_paper_fidelity_acado_v4_independent\runs_3seed --seeds 0,1,2 --device cuda --batch-size 2048 --skip-curve

& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\summarize_capacity_results.py --runs-root .\experiments\falcon_paper_fidelity_acado_v4_independent\runs_3seed
```

仅当 pilot 汇总中的 `capacity_pilot_pass_10_percent` 为 `true` 时运行完整六架构：

```powershell
& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\train_capacity_models.py --mode full --data-dir .\experiments\falcon_paper_fidelity_acado_v4_independent\dnn_dataset --output-root .\experiments\falcon_paper_fidelity_acado_v4_independent\runs_3seed --seeds 0,1,2 --device cuda --batch-size 2048 --skip-curve

& $trainPy .\experiments\falcon_paper_fidelity_acado_v4_independent\summarize_capacity_results.py --runs-root .\experiments\falcon_paper_fidelity_acado_v4_independent\runs_3seed
```
