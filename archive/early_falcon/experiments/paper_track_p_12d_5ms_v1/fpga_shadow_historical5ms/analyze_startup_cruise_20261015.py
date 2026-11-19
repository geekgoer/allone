"""Diagnose startup and sustained tracking lag from existing FPGA flight logs.

Read-only source analysis. Uses all valid flights with no injected delay.
Fixed physical-time intervals: startup [0,2] s, cruise [2,8] s.
Derived fixed-time costs are diagnostics, never replacement SVR labels.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SWEEP = HERE / 'paper_style_sweep/fast_4h_v1'
SOURCE = SWEEP / 'svr_dataset/paper_style_episode_dataset_train_ready.csv'
OUTPUT = HERE / 'audit_20261015/startup_cruise'
WINDOWS = {'startup': (0.0, 2.0), 'cruise': (2.0, 8.0), 'fixed_0_8': (0.0, 8.0)}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def time_mean(time, values, lower, upper):
    """Trapezoidal time average, with linear interpolation only at boundaries."""
    if time[0] > lower or time[-1] < upper:
        raise ValueError('Requested physical-time window is not covered by measurements')
    points = np.concatenate(([lower], time[(time>lower)&(time<upper)], [upper]))
    if values.ndim == 1:
        return float(np.trapz(np.interp(points, time, values), points)/(upper-lower))
    return np.array([time_mean(time, values[:, i], lower, upper) for i in range(values.shape[1])])


def paired_period_changes(frame, field):
    pairs = []
    for key, group in frame.groupby(['model_id', 'scenario_id', 'repeat_index']):
        by_period = group.set_index('period_ms')
        if 30.0 not in by_period.index or 100.0 not in by_period.index:
            continue
        low, high = float(by_period.loc[30., field]), float(by_period.loc[100., field])
        pairs.append(dict(model_id=int(key[0]), scenario_id=key[1], repeat_index=int(key[2]),
                          value_30=low, value_100=high, change_percent=(high/low-1)*100))
    changes = np.array([p['change_percent'] for p in pairs])
    return dict(pairs=len(pairs), mean_change_percent=float(changes.mean()),
                median_change_percent=float(np.median(changes)),
                increasing_fraction=float(np.mean(changes>0)), values=pairs)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(SOURCE)
    metadata = metadata[metadata.actuation_delay_ms == 0].copy()
    results, traces, provenance, skipped = [], [], {}, []
    for _, row in metadata.iterrows():
        key = f'm{int(row.model_id):02d}_{row.scenario_id}_r{int(row.repeat_index):02d}.jsonl'
        path = SWEEP / 'runs' / row.source_directory / 'trial_logs' / key
        provenance[str(path.relative_to(HERE))] = sha256(path)
        with path.open(encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        stamps = np.array([r['state_acquired_us'] for r in records], dtype=np.float64)
        time = (stamps-stamps[0])/1e6
        if time[-1] < 8 or np.any(np.diff(time)<=0):
            skipped.append(dict(record_id=row.record_id, reason='less than 8 seconds or non-increasing timestamps'))
            continue
        actual = np.array([r['paper_psi_actual'] for r in records])
        reference = np.array([r['paper_psi_reference'] for r in records])
        error = actual-reference
        squared = error**2
        speed = np.linalg.norm(actual[:,3:6], axis=1)
        reference_speed = np.linalg.norm(reference[:,3:6], axis=1)
        lag = -error[:,0]
        base = dict(record_id=row.record_id, model_id=int(row.model_id), model_name=row.model_name,
                    period_ms=float(row.period_ms), scenario_id=row.scenario_id, epsilon=float(row.epsilon),
                    repeat_index=int(row.repeat_index), duration_s=float(time[-1]), original_J=float(row.J_paper_style),
                    initial_speed_mps=float(speed[0]), initial_reference_speed_mps=float(reference_speed[0]))
        for name, (lower, upper) in WINDOWS.items():
            mse = time_mean(time, squared, lower, upper)
            mean_lag = time_mean(time, lag, lower, upper)
            base.update({f'{name}_cost':float(mse.sum()),
                f'{name}_J_diagnostic':float(1/(21*mse.sum())),
                f'{name}_x_mse':float(mse[0]), f'{name}_yz_mse':float(mse[1:3].sum()),
                f'{name}_velocity_mse':float(mse[3:6].sum()), f'{name}_quaternion_mse':float(mse[6:].sum()),
                f'{name}_mean_lag_x_m':float(mean_lag),
                f'{name}_speed_mean_mps':time_mean(time,speed,lower,upper),
                f'{name}_reference_speed_mean_mps':time_mean(time,reference_speed,lower,upper),
                # E[lag^2] = E[lag]^2 + Var(lag); no samples are corrected or discarded.
                f'{name}_x_bias_fraction':float(mean_lag**2/mse[0]) if mse[0]>0 else 0.0,
                f'{name}_x_bias_share_total_cost':float(mean_lag**2/mse.sum())})
        for at in (.5, 1., 2., 5., 8.):
            base[f'lag_at_{at:g}s_m'] = float(np.interp(at,time,lag))
        base['cruise_lag_drift_mps'] = float(np.polyfit(time[(time>=2)&(time<=8)],lag[(time>=2)&(time<=8)],1)[0])
        base['startup_integrated_cost_share_0_8'] = float(2*base['startup_cost']/(8*base['fixed_0_8_cost']))
        base['lag_at_1s_fraction_of_cruise_mean'] = base['lag_at_1s_m']/base['cruise_mean_lag_x_m'] if base['cruise_mean_lag_x_m']>0 else None
        results.append(base)
        traces.append(dict(meta=base,time=time,lag=lag,speed=speed,reference_speed=reference_speed))
    frame=pd.DataFrame(results)
    frame.to_csv(OUTPUT/'flight_phase_metrics.csv',index=False)
    focus=frame[np.isclose(frame.epsilon,.8)].copy()
    fields=['initial_speed_mps','initial_reference_speed_mps','lag_at_1s_m','lag_at_2s_m','lag_at_8s_m',
            'startup_mean_lag_x_m','cruise_mean_lag_x_m','cruise_lag_drift_mps',
            'cruise_speed_mean_mps','cruise_reference_speed_mean_mps','cruise_x_bias_fraction',
            'cruise_x_bias_share_total_cost','startup_integrated_cost_share_0_8',
            'original_J','fixed_0_8_J_diagnostic','cruise_J_diagnostic']
    grouped=focus.groupby(['model_id','model_name','period_ms'])[fields].mean().reset_index()
    grouped.to_csv(OUTPUT/'model_period_means_epsilon08.csv',index=False)
    numeric=['initial_speed_mps','initial_reference_speed_mps','lag_at_1s_m','lag_at_2s_m','lag_at_8s_m',
             'cruise_mean_lag_x_m','cruise_lag_drift_mps','cruise_x_bias_fraction','cruise_x_bias_share_total_cost',
             'startup_integrated_cost_share_0_8','lag_at_1s_fraction_of_cruise_mean']
    report=dict(input_csv=str(SOURCE),selected_valid_no_injected_delay=len(metadata),analyzed=len(frame),
        skipped=skipped, physical_time_zero='first logged state_acquired_us per flight',
        time_windows_seconds=WINDOWS, averaging='trapezoidal time average; linear interpolation at window boundaries',
        original_SVR_labels='read only, never replaced',
        diagnostic_J='1/(21*time-averaged sum of 10-dimensional squared state errors), not the original windowed label',
        all_flights_medians={key:float(frame[key].median()) for key in numeric},
        epsilon08_medians={key:float(focus[key].median()) for key in numeric},
        period_effect_epsilon08={key:paired_period_changes(focus,key) for key in
                                ('original_J','fixed_0_8_J_diagnostic','cruise_J_diagnostic')},
        per_model_period_epsilon08=grouped.to_dict('records'),
        source_sha256={SOURCE.name:sha256(SOURCE),**provenance}, script_sha256=sha256(Path(__file__)))
    # Decompose the paired period cost difference into measured component costs.
    paired_costs=[]
    for key,group in focus.groupby(['model_id','scenario_id','repeat_index']):
        indexed=group.set_index('period_ms')
        a,b=indexed.loc[30.],indexed.loc[100.]
        columns=['fixed_0_8_cost','fixed_0_8_x_mse','fixed_0_8_yz_mse','fixed_0_8_velocity_mse',
                 'fixed_0_8_quaternion_mse','startup_cost','cruise_cost']
        paired_costs.append({c:float(a[c]-b[c]) for c in columns})
    differences=pd.DataFrame(paired_costs).mean().to_dict()
    report['mean_cost_30_minus_100_epsilon08']=differences
    report['cost_difference_x_fraction']=differences['fixed_0_8_x_mse']/differences['fixed_0_8_cost']
    report['cost_difference_startup_fraction']=(2/8)*differences['startup_cost']/differences['fixed_0_8_cost']
    (OUTPUT/'phase_report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','per_model_period_epsilon08','period_effect_epsilon08')},indent=2),flush=True)
    print('Paired period effects:',{key:{k:v for k,v in value.items() if k!='values'} for key,value in report['period_effect_epsilon08'].items()},flush=True)
    print(grouped[['model_id','period_ms','lag_at_1s_m','lag_at_2s_m','lag_at_8s_m','cruise_lag_drift_mps',
                   'cruise_speed_mean_mps','cruise_x_bias_share_total_cost']].round(4).to_string(index=False),flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={30:'#0072B2',60:'#D55E00',100:'#009E73'}
    styles={30:'-',60:'--',100:':'}
    grid=np.linspace(0,8,161)
    fig,axes=plt.subplots(2,2,figsize=(12.5,7.7),layout='constrained',sharex=True)
    for row,model in enumerate((1,6)):
        for period in (30,60,100):
            subset=[t for t in traces if t['meta']['model_id']==model and t['meta']['period_ms']==period and np.isclose(t['meta']['epsilon'],.8)]
            for column,variable in enumerate(('lag','speed')):
                samples=np.array([np.interp(grid,t['time'],t[variable]) for t in subset])
                axes[row,column].plot(grid,samples.mean(0),color=colors[period],linestyle=styles[period],label=f'T={period} ms',linewidth=2)
                axes[row,column].fill_between(grid,samples.min(0),samples.max(0),color=colors[period],alpha=.12)
        for column in range(2):
            ax=axes[row,column]
            ax.axvspan(0,2,color='gray',alpha=.1)
            ax.axvline(2,color='gray',linewidth=.8)
            ax.grid(alpha=.2)
            ax.set_title(f'Model {model}: '+('forward position lag' if column==0 else 'actual speed'))
            ax.set_ylabel('Reference x - actual x (m)' if column==0 else 'Speed (m/s)')
            ax.legend(loc='lower right',fontsize=8)
        axes[row,1].axhline(5,color='black',linestyle='-.',linewidth=1,label='Reference speed = 5 m/s')
        axes[row,1].legend(loc='lower right',fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Seconds after first recorded sample')
    fig.suptitle('Existing FPGA flights: startup [0,2] s and cruise [2,8] s\n'
        'epsilon=0.8, zero injected delay; lines are means, colored bands min-max of 4 flights (2 layouts x 2 repeats).')
    fig.savefig(OUTPUT/'startup_lag_and_speed.png',dpi=180)
    plt.close(fig)
    print('Saved:',OUTPUT,flush=True)


if __name__=='__main__':
    main()
