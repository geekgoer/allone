"""Report the entire declared comparison, never just its winning scenarios."""

import argparse
import json
from pathlib import Path

from .decision_dataset import atomic_json


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def number(value):
    return f"{value:.6f}" if value is not None else "—"


def initial_match(directory):
    left,right = (read(directory/p/"initial_state.json",{}) for p in ("static","adaptive"))
    keys = ("position","velocity","quaternion","held_control","matched")
    return bool(left and left.keys()==right.keys() and all(
        all(left[v][k]==right[v][k] for k in keys) for v in left))


def budget_activation(directory):
    rows = [r for r in read(directory/"analyses.json",[]) if r.get("valid")]
    return {"analyses":len(rows),"physical_budget_binds":sum(
        r["analysis_s"]+r["budget_s"] < min(r["actuation_proxy_s"]+.080,
            r["analysis_s"]+r.get("mode_selection",{}).get("delay_limit_s",float("inf"))) for r in rows),
        "mode_wait_limit_binds":sum(r["analysis_s"]+r.get("mode_selection",{}).get("delay_limit_s",float("inf"))
            < min(r["actuation_proxy_s"]+.080,r["analysis_s"]+r["budget_s"]) for r in rows)}


def build(root, output):
    setup = read(root/"setup.json",{})
    calibration = read(root/"calibration_results.json",[])
    reused_search = setup.get("source_search") if not calibration else None
    if reused_search:
        calibration = read(Path(reused_search)/"calibration_results.json",[])
    selection = read(root/"selection.json",{})
    discovery = read(root/"discovery_summary.json",{})
    confirmation = read(root/"confirmation_summary.json",{})
    validation = read(root/"baseline_validation.json",[])
    collision_valid = setup.get("collision_observation") == "simGetCollisionInfo_changed_nonzero_stamp_v1"
    lines = [f"# {setup.get('vehicles','多')}机：静态性能搜索与状态框架的配对结果","",
        "这是CPU有限实验，使用经验名义/尾部预测，不是WCET、CBF安全证明或完整FALCON复现。",
        f"运行目录：`{root.resolve()}`。版本：`{setup.get('version','尚无setup')}`。","",
        ("碰撞来源：专用API及新事件时间戳。" if collision_valid else
            "**碰撞统计未通过来源验证：旧零记录不能证明无碰撞，不据此确认安全优势。**"),"",
        "## 正常工况的静态候选搜索","",
        f"已完成候选：{len(calibration)}；通过全部可比条件：{sum(r['score'] is not None for r in calibration)}。",
        "J为原式形式的完整窗口状态跟踪评分（Wx=I、公共20ms网格），不是DNN对MPC输出的离线MSE。",
        "恢复日志覆盖不足的候选不等同于观察到控制错误；完整原始报告保留。","",
        "| 候选 | 完整可比 | 总J | 中止/其他未通过条件 |","|---|---|---:|---|"]
    for row in calibration:
        report = row["report"]
        reason = report["abort"]["kind"] if report["abort"] else report["error"]
        if not reason and not report["accepted"]:
            reasons = []
            if not report["recovery_passed"]:
                reasons.append("恢复状态或日志覆盖未通过")
            if not report["clock_valid"]:
                reasons.append("物理时钟未通过")
            reason = "/".join(reasons) or "其他条件未通过"
        lines.append(f"| {row['name']} | {report['accepted']} | {number(report['total_J'])} | {reason or '无'} |")
    if reused_search:
        source_screen = read(root/"screening_source.json",{})
        lines += ["",f"上述初次候选来自保留的搜索：`{reused_search}`；本批没有重飞全部候选。",
            f"此前已复核候选被排除，不重复等成功：{', '.join(source_screen.get('excluded_already_screened',[]))}。"]
    if validation:
        lines += ["","## 静态候选重复复核","",
            "本批三个候选各额外运行两次；三次均通过才可选型，按三次平均J排序。","",
            "| 候选 | 初次通过 | 两次额外通过 | 三次平均J | 可参与最终选型 |","|---|---|---|---:|---|"]
        for row in validation:
            passed = "/".join(str(r["accepted"]) for r in row["validation_reports"])
            lines.append(f"| {row['name']} | {row['report']['accepted']} | {passed} | {number(row['score'])} | {row['all_runs_passed']} |")
    if selection.get("candidate") and selection.get("adaptive_nominal"):
        static = selection["candidate"]["report"]
        adaptive = selection["adaptive_nominal"]
        static_score = selection["candidate"]["score"]
        ratio = adaptive["total_J"]/static_score if adaptive["accepted"] and static["accepted"] and static_score else None
        lines += ["",f"选中静态候选：`{selection['candidate']['name']}`；选择状态：`{selection['status']}`。",
            f"静态选型分数J={number(static_score)}；动态正常工况J={number(adaptive['total_J'])}；比率={number(ratio)}。",
            "有重复复核时，静态分数是三轮均值，动态此处是一轮观察；该比率不是配对效应估计。",
            "这是观测值，不是统计非劣或总体最优证明。"]
        if selection.get("source_static_search"):
            source = Path(selection["source_static_search"])
            previous = read(source.parent/"calibration_results.json",[])
            lines += [f"初次候选来自先前{len(previous)}个配置的冻结搜索：`{source}`。当前选择与追加复核记录均保留。"]
        activation = budget_activation(root/"calibration/adaptive_nominal")
        lines += [f"动态正常复核：有效安全分析{activation['analyses']}次，物理B先于基本更新及模式等待上限生效{activation['physical_budget_binds']}次，模式等待上限先行生效{activation['mode_wait_limit_binds']}次。",
            "若物理B激活为0，正常工况只能说明集成/性能观察，不能用它证明物理B本身带来的安全收益。"]
    elif selection:
        lines += ["",f"选型尚未通过：`{selection.get('status','UNKNOWN')}`。未启动后续挑战不等于安全完成。"]
    verified_wins = 0
    lines += ["","## 预声明扰动配对","",
        f"发现集完成：{discovery.get('complete',False)}；已完成配对：{len(discovery.get('pairs',[]))}/{len(setup.get('scenarios',[])) or 16}。","",
        "| 场景 | 初态契约一致 | 静态碰撞 | 静态完成 | 动态完成 | 动态J | 干净安全优势 |","|---|---|---:|---|---|---:|---|"]
    for pair in discovery.get("pairs",[]):
        name = pair["scenario"]["id"]
        match = bool(pair.get("initial_state_match",False) and initial_match(root/"discovery"/name))
        win = bool(collision_valid and pair["clean_safety_win"] and match)
        verified_wins += win
        a,b = pair["static"],pair["adaptive"]
        lines.append(f"| {name} | {match} | {a['collisions']} | {a['accepted']} | {b['accepted']} | {number(b['total_J'])} | {win} |")
    lines += ["",f"核对初态后的干净安全优势数：**{verified_wins}**。",
        "静态碰撞而动态中止，不算安全任务完成；基线撞毁后的不完整J不与完整动态J作大小比较。",
        f"相邻扰动确认/消融完成：{confirmation.get('complete',False)}；记录数：{len(confirmation.get('rows',[]))}。",
        "所有失败、预测拒绝、实际时序违规均保留，不能只挑选有利场景。","",
        "## 当前证据边界","",
        "只有发现集完成且存在独立确认，才能据此讨论可重复的安全优势。没有优势时应报告负结果。",
        "模式仅在任务边界变化，抢占仅在段边界；CPU时延不适用于FPGA。B仍是启发式。",
        "上述基线是受限模型/周期族的经验静态搜索，没有SVM和全硬件搜索，不能写成完整FALCON的全局最优解。"]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text("\n".join(lines)+"\n",encoding="utf-8")
    summary = {"root":str(root.resolve()),"calibration_candidates":len(calibration),
        "reused_initial_search":reused_search,
        "collision_observation_valid":collision_valid,"baseline_validation_candidates":len(validation),
        "repeatedly_valid_static_candidates":sum(r["all_runs_passed"] for r in validation),
        "selection_status":selection.get("status"),"discovery_complete":discovery.get("complete",False),
        "pairs":len(discovery.get("pairs",[])),"verified_clean_safety_wins":verified_wins,
        "confirmation_complete":confirmation.get("complete",False),"FPGA_verified":False}
    summary["normal_budget_activation"] = budget_activation(root/"calibration/adaptive_nominal")
    atomic_json(output.with_suffix(".json"),summary)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    build(args.root,args.output)
