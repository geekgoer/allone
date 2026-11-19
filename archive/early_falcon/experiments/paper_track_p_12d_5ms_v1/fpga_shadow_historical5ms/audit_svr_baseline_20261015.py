"""Read-only audit of the frozen SVR data/model and conditional trend evidence."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / 'paper_style_sweep/fast_4h_v1/svr_dataset'
RUN = HERE / 'svr_runs/fast_4h_v1_seed42_small_epsilon'
OUT = HERE / 'audit_20261015'


def predict(payload, frame):
    x = frame[['c_inverse_beta', 'period_ms', 'R_max_ms', 'epsilon']].to_numpy(dtype=float)
    x[:, :3] /= np.asarray(payload['feature_maximum_for_c_T_R'])
    return payload['model'].predict(x)


def change_summary(frame, column, fixed, increasing_good):
    pairs = []
    for key, group in frame.groupby(fixed):
        means = group.groupby(column)['J_paper_style'].mean().sort_index()
        if len(means) < 2:
            continue
        low, high = float(means.iloc[0]), float(means.iloc[-1])
        change = (high-low)/low*100
        rho = float(spearmanr(means.index, means.values).statistic)
        pairs.append(dict(group=str(key), low=low, high=high, relative_change_percent=change, spearman=rho))
    values = np.array([p['relative_change_percent'] for p in pairs])
    return dict(groups=len(pairs), expected_direction_fraction=float(np.mean(values > 0 if increasing_good else values < 0)),
                mean_change_percent=float(values.mean()), median_change_percent=float(np.median(values)),
                median_within_group_spearman=float(np.median([p['spearman'] for p in pairs])), pairs=pairs)


def main():
    OUT.mkdir(exist_ok=True)
    all_rows = pd.read_csv(DATA / 'paper_style_episode_dataset_all.csv')
    data = pd.read_csv(DATA / 'paper_style_episode_dataset_train_ready.csv')
    payload = joblib.load(RUN / 'svr_model_all_data.joblib')
    data['svr_prediction'] = predict(payload, data)
    report = dict(total=len(all_rows), valid=len(data), invalid=len(all_rows)-len(data),
        ranges={k:[float(data[k].min()), float(data[k].max())] for k in
                ('c_inverse_beta','period_ms','R_max_ms','epsilon','J_paper_style')},
        holdout_metrics=json.loads((RUN / 'metrics.json').read_text())['test_metrics'])
    models = data.groupby(['model_id','model_name','c_inverse_beta'], as_index=False).agg(
        n=('J_paper_style','size'), J_mean=('J_paper_style','mean'), J_std=('J_paper_style','std'), R_mean=('R_max_ms','mean'))
    models = models.sort_values('c_inverse_beta')
    models.to_csv(OUT / 'model_means.csv', index=False)
    report['models'] = models.to_dict('records')
    report['periods'] = data.groupby('period_ms')['J_paper_style'].agg(['count','mean','std']).reset_index().to_dict('records')
    report['difficulties'] = data.groupby('epsilon')['J_paper_style'].agg(['count','mean','std']).reset_index().to_dict('records')
    report['delay_effect'] = change_summary(data, 'actuation_delay_ms', ['model_id','period_ms','scenario_id','repeat_index'], False)
    report['period_effect_delay_zero'] = change_summary(data[data.actuation_delay_ms==0], 'period_ms',
                                                       ['model_id','scenario_id','repeat_index'], False)
    report['capability_effect'] = change_summary(data, 'c_inverse_beta',
                                                ['period_ms','actuation_delay_ms','scenario_id','repeat_index'], True)
    focus = data[np.isclose(data.epsilon,.8)]
    report['epsilon08_effects'] = dict(
        delay=change_summary(focus, 'actuation_delay_ms', ['model_id','period_ms','scenario_id','repeat_index'], False),
        period_delay_zero=change_summary(focus[focus.actuation_delay_ms==0], 'period_ms', ['model_id','scenario_id','repeat_index'], False),
        capability=change_summary(focus, 'c_inverse_beta', ['period_ms','actuation_delay_ms','scenario_id','repeat_index'], True))
    repeats = data.groupby(['model_id','period_ms','actuation_delay_ms','scenario_id'])['J_paper_style'].agg(['mean','std','count'])
    report['repeat_coefficient_of_variation_median'] = float((repeats['std']/repeats['mean']).median())
    bad = all_rows[~all_rows.valid_for_svr.astype(str).str.lower().eq('true')]
    report['excluded'] = bad[['model_id','scenario_id','period_ms','actuation_delay_ms','deadline_miss_count','fallback_count','error']].fillna('').to_dict('records')

    # Check the raw logged state errors and aggregate J, not just the summary CSV.
    raw_trials = {}
    max_error_difference = 0.0
    raw_dir = HERE / 'paper_style_sweep/fast_4h_v1/runs'
    for directory in sorted(raw_dir.iterdir()):
        if not directory.is_dir():
            continue
        with (directory / 'fpga_closed_loop_trials.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                key = (directory.name, int(row['model_id']), row['scenario_id'], int(row['repeat_index']))
                record = raw_trials.setdefault(key, dict(sse=[], components=[], timestamps=[], clips=0))
                a, b = np.array(row['paper_psi_actual']), np.array(row['paper_psi_reference'])
                squared = (a-b)**2
                max_error_difference = max(max_error_difference, float(np.max(np.abs(squared-row['paper_state_error_squared']))))
                record['sse'].append(float(squared.sum()))
                record['components'].append([squared[:3].sum(), squared[3:6].sum(), squared[6:].sum()])
                record['timestamps'].append(row['state_acquired_us'])
                record['clips'] += int(any(row['clip_flags']))
        print('Raw audit:', directory.name, flush=True)
    verification = []
    for _, row in data.iterrows():
        key = (row.source_directory, int(row.model_id), row.scenario_id, int(row.repeat_index))
        record = raw_trials[key]
        sse = np.array(record['sse'])
        used = len(sse)//21*21
        calculated_j = 1/(sse[:used].reshape(-1,21).sum(1).mean())
        components = np.array(record['components'])[:used].mean(0)
        verification.append(dict(record_id=row.record_id, period_ms=row.period_ms, J_recorded=row.J_paper_style,
            J_recomputed=calculated_j, J_all_samples=1/(21*sse.mean()),
            position_mse_sum=components[0], velocity_mse_sum=components[1], quaternion_mse_sum=components[2],
            position_rmse_m=float(np.sqrt(components[0])), velocity_rmse_mps=float(np.sqrt(components[1])),
            median_actual_period_ms=float(np.median(np.diff(record['timestamps']))/1000),
            discarded_fraction=(len(sse)-used)/len(sse), clipped_cycle_fraction=record['clips']/len(sse)))
    verification = pd.DataFrame(verification)
    verification.to_csv(OUT / 'raw_metric_verification.csv', index=False)
    matched=data.merge(verification,on='record_id',suffixes=('','_raw'))
    matched=matched[np.isclose(matched.epsilon,.8)&(matched.actuation_delay_ms==0)]
    period_all_sample_changes=[]
    for _, group in matched.groupby(['model_id','scenario_id','repeat_index']):
        by_period=group.set_index('period_ms')
        period_all_sample_changes.append((by_period.loc[100,'J_all_samples']/by_period.loc[30,'J_all_samples']-1)*100)
    report['raw_checks'] = dict(max_state_squared_error_mismatch=max_error_difference,
        max_J_absolute_mismatch=float(np.max(np.abs(verification.J_recorded-verification.J_recomputed))),
        median_position_rmse_m=float(verification.position_rmse_m.median()),
        median_velocity_rmse_mps=float(verification.velocity_rmse_mps.median()),
        mean_components=verification[['position_mse_sum','velocity_mse_sum','quaternion_mse_sum']].mean().to_dict(),
        maximum_discarded_fraction=float(verification.discarded_fraction.max()),
        median_relative_J_change_using_all_samples=float(np.median((verification.J_all_samples/verification.J_recorded-1)*100)),
        max_abs_relative_J_change_using_all_samples=float(np.max(np.abs(verification.J_all_samples/verification.J_recorded-1))*100),
        period_30_to_100_mean_change_percent_all_samples=float(np.mean(period_all_sample_changes)),
        period_30_to_100_decreasing_fraction_all_samples=float(np.mean(np.array(period_all_sample_changes)<0)),
        median_actual_period_by_T=verification.groupby('period_ms').median_actual_period_ms.median().to_dict())

    # Model derivatives are conditional slices, not measured causal effects.
    derivatives = {}
    grid = pd.DataFrame(np.array(np.meshgrid(np.linspace(focus.c_inverse_beta.min(),focus.c_inverse_beta.max(),13),
        [30.,60.,100.],np.linspace(focus.R_max_ms.min(),focus.R_max_ms.max(),17),[.8],indexing='ij')).reshape(4,-1).T,
        columns=['c_inverse_beta','period_ms','R_max_ms','epsilon'])
    for col, expected_sign in [('c_inverse_beta',1),('period_ms',-1),('R_max_ms',-1)]:
        offset=(grid[col].max()-grid[col].min())*.001
        plus,minus=grid.copy(),grid.copy()
        plus[col]+=offset;minus[col]-=offset
        delta=predict(payload,plus)-predict(payload,minus)
        derivatives[col]=dict(expected_direction_fraction=float(np.mean(delta*expected_sign>0)),
                              median_derivative=float(np.median(delta/(2*offset))))
    report['model_grid_derivatives_epsilon08_unrestricted_box'] = derivatives
    report['model_derivative_warning']='Box includes unsampled combinations; derivative signs do not establish measured effects.'
    data.to_csv(OUT / 'observations_and_predictions.csv', index=False)
    (OUT / 'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('excluded','delay_effect','capability_effect','period_effect_delay_zero','epsilon08_effects')},indent=2),flush=True)
    print('epsilon08_effects:',{k:{a:b for a,b in v.items() if a!='pairs'} for k,v in report['epsilon08_effects'].items()},flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    # Controlled slices with raw repetitions visible; no extrapolation or curve fitting.
    slices=[(focus[(focus.period_ms==60)&(focus.actuation_delay_ms==0)],'c_inverse_beta','Capability c (60 ms, no added delay)'),
            (focus[focus.actuation_delay_ms==0],'period_ms','Period T (ms), no added delay'),
            (focus[focus.period_ms==60],'actuation_delay_ms','Added delay (ms), T=60 ms')]
    for ax,(frame,column,label) in zip(axes,slices):
        if column=='c_inverse_beta':
            group_columns=['scenario_id']
        else:
            group_columns=['model_id']
        for group,sub in frame.groupby(group_columns):
            ax.scatter(sub[column],sub.J_paper_style*1000,s=13,alpha=.35)
            means=sub.groupby(column).J_paper_style.mean()
            label_group=group[0] if isinstance(group,tuple) else group
            ax.plot(means.index,means.values*1000,'o-',label=str(label_group),markersize=4)
        ax.set(xlabel=label,ylabel='Measured J x 1000');ax.grid(alpha=.2)
        ax.legend(title='Scenario' if column=='c_inverse_beta' else 'Model ID',fontsize=7)
    fig.suptitle('Frozen FALCON baseline: measured conditional trends, environment epsilon=0.8\nLines connect measured means; model and scenario effects remain distinct.')
    fig.savefig(OUT/'measured_trends.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
