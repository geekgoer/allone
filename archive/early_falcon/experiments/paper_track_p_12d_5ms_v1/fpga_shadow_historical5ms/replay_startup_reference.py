"""Compare two evaluation references on identical recorded flights; no hardware use."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from environment import GateEnvironment
from gate_route import ContinuousGateRoute
from startup_reference import StartupPolylineReference, reference_contract
from analyze_startup_cruise_20261015 import time_mean


def window_j(squared):
    used=len(squared)//21*21
    return float(1/np.mean(np.sum(squared[:used].reshape(-1,21,10),axis=(1,2))))


def main():
    source=HERE/'paper_style_sweep/fast_4h_v1'
    output=HERE/'audit_20261015/startup_reference_replay'
    output.mkdir(parents=True,exist_ok=True)
    benchmark=json.loads((HERE.parent/'physical_benchmark_v1/benchmark_config.json').read_text())
    gate_cfg=benchmark['gate']
    cases=pd.read_csv(source/'svr_dataset/paper_style_episode_dataset_train_ready.csv')
    cases=cases[cases.actuation_delay_ms==0]
    scenarios={s['id']:s for s in benchmark['scenarios']}
    result=[]
    illustrations=[]
    for _,case in cases.iterrows():
        scenario=scenarios[case.scenario_id]
        path=source/'runs'/case.source_directory/'trial_logs'/f'm{int(case.model_id):02d}_{case.scenario_id}_r{int(case.repeat_index):02d}.jsonl'
        with path.open(encoding='utf-8') as handle:
            rows=[json.loads(line) for line in handle]
        stamps=np.array([r['state_acquired_us'] for r in rows])
        t=(stamps-stamps[0])/1e6
        actual=np.array([r['paper_psi_actual'] for r in rows])
        original=np.array([r['paper_psi_reference'] for r in rows])
        gates=GateEnvironment(epsilon=scenario['epsilon'],seed=scenario['seed'],
            **{k:gate_cfg[k] for k in ('base_y','base_z','max_z')},gate_count=gate_cfg['count'],
            x_start=gate_cfg['x_start'],x_spacing=gate_cfg['x_spacing']).generate()
        route=ContinuousGateRoute(gates,actual[0,:3],start_time=0,
            reference_speed=benchmark['mpc']['reference_speed'],
            **{k:gate_cfg[k] for k in ('approach_distance','exit_distance','completion_radius')})
        ramp=StartupPolylineReference(route.waypoints,actual[0,3:6],cruise_speed=5.,duration=2.)
        refs=[ramp.at(float(time),orientation=original[i,6:]) for i,time in enumerate(t)]
        smooth=np.array([np.concatenate((r.position,r.velocity,r.orientation)) for r in refs])
        old_squared=(actual-original)**2
        new_squared=(actual-smooth)**2
        old_j,new_j=window_j(old_squared),window_j(new_squared)
        np.testing.assert_allclose(old_j,float(case.J_paper_style),rtol=1e-10,atol=1e-12)
        result.append(dict(record_id=case.record_id,model_id=int(case.model_id),period_ms=float(case.period_ms),
            scenario_id=case.scenario_id,epsilon=float(case.epsilon),repeat_index=int(case.repeat_index),
            original_J=old_j, smooth_reference_J_diagnostic=new_j,
            old_fixed_0_8_cost=float(time_mean(t,old_squared.sum(1),0,8)),
            smooth_fixed_0_8_cost=float(time_mean(t,new_squared.sum(1),0,8)),
            smooth_cruise_x_mean_error_m=float(time_mean(t,actual[:,0]-smooth[:,0],2,8)),
            startup_duration_s=2.))
        if case.model_id in (1,6) and case.period_ms==60 and case.scenario_id=='S02' and case.repeat_index==0:
            illustrations.append(dict(model=int(case.model_id),time=t,actual=actual,constant=original,smooth=smooth))
    frame=pd.DataFrame(result)
    frame.to_csv(output/'same_flights_reference_comparison.csv',index=False)
    focus=frame[np.isclose(frame.epsilon,.8)]
    pairs=[]
    for key,group in focus.groupby(['model_id','scenario_id','repeat_index']):
        g=group.set_index('period_ms')
        pairs.append(dict(model_id=int(key[0]),scenario_id=key[1],repeat_index=int(key[2]),
            original_J_change_percent=(g.loc[100,'original_J']/g.loc[30,'original_J']-1)*100,
            smooth_J_change_percent=(g.loc[100,'smooth_reference_J_diagnostic']/g.loc[30,'smooth_reference_J_diagnostic']-1)*100))
    report=dict(status='offline_evaluation_only', flights=len(frame), reference=reference_contract('smooth-start',2),
        definition='Identical recorded actual states; initial position/velocity from first recorded sample; same waypoint geometry and orientation reference.',
        limitation='This does not predict a changed flight or establish a causal improvement in control; original SVR labels are unchanged.',
        mean_original_J=float(focus.original_J.mean()),mean_smooth_J=float(focus.smooth_reference_J_diagnostic.mean()),
        period_30_to_100_original_mean_change_percent=float(np.mean([p['original_J_change_percent'] for p in pairs])),
        period_30_to_100_smooth_mean_change_percent=float(np.mean([p['smooth_J_change_percent'] for p in pairs])),
        period_smooth_increasing_fraction=float(np.mean([p['smooth_J_change_percent']>0 for p in pairs])),
        paired_changes=pairs,
        model_means_T60_epsilon08=focus[focus.period_ms==60].groupby('model_id')[['original_J','smooth_reference_J_diagnostic']].mean().reset_index().to_dict('records'))
    (output/'reference_replay_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.3),layout='constrained')
    for ax,record in zip(axes,sorted(illustrations,key=lambda r:r['model'])):
        selected=record['time']<=8
        for key,color,style,label in [('actual','#0072B2','-','Recorded actual position'),
                ('constant','#D55E00','--','Original 5 m/s reference'),('smooth','#009E73',':','Initial-state-matched reference')]:
            ax.plot(record['time'][selected],record[key][selected,0],color=color,linestyle=style,label=label,linewidth=2)
        ax.axvspan(0,2,color='gray',alpha=.1)
        ax.set(xlabel='Time (s)',ylabel='Forward position x (m)',title=f'Model {record["model"]}, S02, T=60 ms, repeat 0')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Reference-definition check on the same recorded flights\nNew reference is a fixed 2-second polynomial startup; no shift fitted to observed tracking errors.')
    fig.savefig(output/'reference_definition_comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
