"""Analyze the completed deep-narrow architecture sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "controller_architecture_study"
SOURCE = STUDY / "results" / "deep_narrow_v1"
OUT = SOURCE / "analysis_report_20260929"


def read_run(folder: Path) -> dict:
    metrics = json.loads((folder / "offline_metrics.json").read_text(encoding="utf-8"))
    complexity = json.loads((folder / "complexity.json").read_text(encoding="utf-8"))
    with (folder / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    best = min(history, key=lambda row: float(row["validation_normalized_mse"]))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    test = metrics["test"]
    val = metrics["validation"]
    return {
        "model_id": folder.name,
        "model_type": config["model_type"],
        "layers": int(config["hidden_layers"]),
        "width": int(config["hidden_width"]),
        "parameters": int(complexity["parameter_count"]),
        "macs": int(complexity["mac_count"]),
        "best_epoch": int(best["epoch"]),
        "epochs_completed": int(history[-1]["epoch"]),
        "train_mse_at_best": float(best["train_normalized_mse"]),
        "val_mse_at_best": float(best["validation_normalized_mse"]),
        "test_normalized_mse": float(test["normalized_mse"]),
        "test_physical_mse": float(test["physical_mse"]),
        "test_rmse": float(test["physical_rmse"]),
        "test_p95": float(test["p95_error"]),
        "test_p99": float(test["p99_error"]),
        "val_physical_mse": float(val["physical_mse"]),
        "history_path": str(folder / "train_log.csv"),
    }


def load_rows() -> list[dict]:
    rows = []
    for folder in sorted(SOURCE.iterdir()):
        if not folder.is_dir() or not (folder / "offline_metrics.json").exists():
            continue
        rows.append(read_run(folder))
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 completed runs, found {len(rows)}")
    return rows


def write_csv(rows: list[dict]) -> None:
    with (OUT / "deep_narrow_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict]) -> None:
    colors = {"plain_mlp": "#0072B2", "residual_mlp": "#D55E00"}
    markers = {"plain_mlp": "o", "residual_mlp": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")

    ax = axes[0, 0]
    for row in rows:
        ax.scatter(row["parameters"], row["test_physical_mse"], c=colors[row["model_type"]],
                   marker=markers[row["model_type"]], s=64)
        prefix = "P" if row["model_type"] == "plain_mlp" else "R"
        ax.annotate(f'{prefix} {row["layers"]}x{row["width"]}', (row["parameters"], row["test_physical_mse"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="Parameters", ylabel="Test physical MSE",
           title="Accuracy versus model size")

    ax = axes[0, 1]
    for row in rows:
        ax.scatter(row["macs"], row["test_rmse"], c=colors[row["model_type"]],
                   marker=markers[row["model_type"]], s=64)
        prefix = "P" if row["model_type"] == "plain_mlp" else "R"
        ax.annotate(f'{prefix} {row["layers"]}x{row["width"]}', (row["macs"], row["test_rmse"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="MACs per inference", ylabel="Test physical RMSE",
           title="RMSE versus compute")

    ax = axes[1, 0]
    ordered = sorted(rows, key=lambda r: r["test_normalized_mse"])
    labels = [f'{"P" if r["model_type"] == "plain_mlp" else "R"} {r["layers"]}x{r["width"]}' for r in ordered]
    values = [r["test_normalized_mse"] for r in ordered]
    bar_colors = [colors[r["model_type"]] for r in ordered]
    ax.barh(range(len(ordered)), values, color=bar_colors)
    ax.set_xscale("log")
    ax.set_yticks(range(len(ordered)), labels, fontsize=8)
    ax.set_xlabel("Test normalized MSE")
    ax.set_title("All 12 models, lower is better")

    ax = axes[1, 1]
    for row in rows:
        gap = row["test_normalized_mse"] / row["train_mse_at_best"]
        ax.scatter(row["parameters"], gap, c=colors[row["model_type"]],
                   marker=markers[row["model_type"]], s=64)
        ax.annotate(f'{row["layers"]}x{row["width"]}', (row["parameters"], gap),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color="#777777", lw=0.8)
    ax.set(xscale="log", yscale="log", xlabel="Parameters", ylabel="Test / train normalized MSE",
           title="Generalization/optimization diagnostic")

    for ax in axes.flat:
        ax.grid(alpha=0.2, which="both")
    fig.suptitle("Deep-narrow controller architecture sweep (seed 0)", fontsize=15)
    fig.text(0.01, 0.01, "Same 12D dataset and test split; circles=plain MLP, squares=residual MLP. No smoothing.", fontsize=9)
    fig.savefig(OUT / "deep_narrow_overview.png", dpi=180, facecolor="white")
    plt.close(fig)


def write_report(rows: list[dict]) -> None:
    best = min(rows, key=lambda row: row["test_physical_mse"])
    smallest = min(rows, key=lambda row: row["parameters"])
    report = [
        "# Deep-narrow 架构实验结果报告",
        "",
        "## 实验范围",
        "",
        "本报告分析 `deep_narrow_v1` 中已完成的 12 个模型：普通 MLP 与残差 MLP，各覆盖 12x128、16x128、22x64、22x128、22x256、34x128。所有模型使用相同的 12D 数据集、相同测试集和 seed=0；指标是 DNN 相对 MPC 标签的离线误差。",
        "",
        "图表：`deep_narrow_overview.png`；原始汇总表：`deep_narrow_metrics.csv`。",
        "",
        "## 主要结果",
        "",
        f"测试集物理 MSE 最低的是 `{best['model_id']}`：{best['test_physical_mse']:.6g}，RMSE={best['test_rmse']:.6g}，P95 绝对误差={best['test_p95']:.6g}，参数量={best['parameters']:,}。",
        f"参数量最小的是 `{smallest['model_id']}`：{smallest['parameters']:,} 参数，测试物理 MSE={smallest['test_physical_mse']:.6g}。",
        "",
        "从这 12 次结果看，误差没有随着深度或宽度单调下降。当前任务中，增加模型规模没有自动带来更好的 MPC 拟合；这与教师映射较简单、以及深层模型在固定学习率和有限 epoch 下优化不充分两种因素都相容。",
        "",
        "## 普通与残差结构",
        "",
        "在相同层数和宽度下，普通与残差模型的参数量和 MAC 数基本相同；残差结构额外增加逐元素加法和分支缩放。残差网络是否更好，应按相同尺寸逐对比较，而不能只比较某个残差模型和另一个尺寸的普通模型。",
        "",
    ]
    for layers, width in sorted({(r["layers"], r["width"]) for r in rows}):
        pair = [r for r in rows if r["layers"] == layers and r["width"] == width]
        if len(pair) != 2:
            continue
        plain = next(r for r in pair if r["model_type"] == "plain_mlp")
        residual = next(r for r in pair if r["model_type"] == "residual_mlp")
        ratio = residual["test_physical_mse"] / plain["test_physical_mse"]
        better = "残差更低" if ratio < 1 else "普通更低"
        report.append(f"- `{layers}x{width}`：普通={plain['test_physical_mse']:.6g}，残差={residual['test_physical_mse']:.6g}，残差/普通={ratio:.3f}，{better}。")
    report += [
        "",
        "## 训练是否充分",
        "",
        "部分模型在达到 500 epoch 上限时仍未触发早停，尤其需要关注残差 12x128、16x128、22x128、以及普通 16x128 和 22x64。这说明这些模型的当前结果包含训练预算因素，不能直接解释为它们的最终容量上限。相反，已经早停的模型更接近当前训练协议下的稳定点。",
        "",
        "建议查看图中右下角的 test/train normalized MSE 比值：比值很大通常说明训练误差已经较低而测试误差较高；比值接近 1 但误差仍偏高，则更像优化或任务表示能力不足。这里没有把这两个原因强行归结为过拟合。",
        "",
        "## 结论与下一步",
        "",
        "这批实验足以说明：在当前 MPC 教师、12D 输入和数据分布下，‘更深/更宽必然更好’不成立；小模型可能已经覆盖了主要控制规律。它还不足以证明全局最优架构，因为每个配置只有一个 seed，且不同模型的有效训练轮次不同。",
        "",
        "下一步应优先对当前排名靠前的 2 至 3 个结构补 seed=1/2，并给未充分收敛的大模型做独立的低学习率或更长预算诊断。只有在多个 seed、独立环境测试和闭环指标都一致时，才适合把某个结构定为最终候选。",
        "",
        "## 可视化说明",
        "",
        "图中圆点表示普通 MLP，方块表示残差 MLP；横纵轴采用对数尺度以同时显示小模型和大模型。所有点都保留，没有平滑、删点或人为缩放。",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_csv(rows)
    plot_metrics(rows)
    write_report(rows)
    print(json.dumps({"output": str(OUT), "models": len(rows), "best_model": min(rows, key=lambda r: r["test_physical_mse"])["model_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
