"""Resume a 24-flight reference comparison: models 1/6, S02, 3 periods, 2 profiles, 2 repeats."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-fpga-actuation',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--output-root',type=Path,default=HERE/'paper_style_sweep/startup_reference_comparison_v1')
    args=parser.parse_args()
    if not args.dry_run and not args.enable_fpga_actuation:
        raise SystemExit('Add --enable-fpga-actuation to run flights, or --dry-run to inspect commands.')
    root=args.output_root.resolve()
    plan=[dict(period_ms=t,profile=p) for t in (30,60,100)
          for p in (('smooth-start','constant') if t==60 else ('constant','smooth-start'))]
    signature=dict(models=[1,6],scenario='S02',repeats=2,startup_duration_s=2.,delay_ms=0.,plan=plan)
    if not args.dry_run:
        root.mkdir(parents=True,exist_ok=True)
        manifest_path=root/'comparison_manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=signature:
            raise ValueError('Existing comparison configuration differs; use a new output root')
        manifest_path.write_text(json.dumps(signature,indent=2),encoding='utf-8')
    run_results=[]
    for run in plan:
        directory=root/f'T{run["period_ms"]:03d}ms_{run["profile"]}'
        summary_path=directory/'fpga_closed_loop_summary.json'
        previous=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if previous.get('episodes_requested')==4 and previous.get('episodes_completed')==4:
            print('SKIP completed:',directory.name,flush=True)
        else:
            command=[sys.executable,str(HERE/'run_fpga_closed_loop.py'),'--enable-fpga-actuation',
                '--fpga-config',str(HERE/'fpga_current5ms_config.json'),'--scenario-set','S02','--model-ids','1,6',
                '--repeats','2','--period-ms',str(run['period_ms']),'--actuation-delay-ms','0',
                '--reference-profile',run['profile'],'--startup-duration-s','2','--output-dir',str(directory)]
            if (directory/'closed_loop_manifest.json').exists():
                command.append('--resume')
            print(subprocess.list2cmdline(command),flush=True)
            if not args.dry_run:
                subprocess.run(command,cwd=ROOT,check=True)
        if not args.dry_run:
            summary=json.loads(summary_path.read_text())
            for trial in summary['trials']:
                clean=bool(trial['completed'] and not trial['collision'] and not trial['error'] and
                           trial['fallback_count']==trial['deadline_miss_count']==trial['timeout_count']==0)
                run_results.append(dict(period_ms=run['period_ms'],reference_profile=run['profile'],
                    model_id=trial['model_id'],repeat_index=trial['repeat_index'],completed=trial['completed'],
                    clean_fpga_episode=clean,J_paper_style=trial['J_paper_style'],
                    R_max_ms=trial.get('measured_R_max_ms'),error=trial['error']))
            report=dict(episodes_requested=24,episodes_recorded=len(run_results),
                        clean_fpga_episodes=sum(r['clean_fpga_episode'] for r in run_results),trials=run_results)
            (root/'comparison_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Dry run complete.' if args.dry_run else f'Comparison saved: {root / "comparison_summary.json"}',flush=True)


if __name__=='__main__':
    main()
