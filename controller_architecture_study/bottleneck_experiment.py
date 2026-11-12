"""Seeded PS/PL sensitivity study with a small, fully enumerated oracle.

No AirSim, torch or FPGA required. Supplied controller qualifications and service
times are SYNTHETIC ASSUMPTIONS. No collision, QoC or board-speed claims result.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from .bottleneck_joint import (Job,Loop,Mode,Option,Platform,Scenario,exhaustive,heuristic,
                               simulate,witness,make_jobs)


def random_case(seed, n, regime, full_grid=False):
    rng=random.Random(seed)
    settings={
        'balanced':dict(prep=.4,dispatch=.05,post=.2,switch=.04),
        'ps_pressure':dict(prep=2.8,dispatch=.15,post=1.,switch=.04),
        'pl_pressure':dict(prep=.3,dispatch=.04,post=.15,switch=.05),
        'switch_pressure':dict(prep=.3,dispatch=.4,post=.15,switch=.8),
    }
    p=Platform(**settings[regime],decision=.5,initial_pl_block=rng.choice((0.,1.,3.)))
    loops=[]
    for i in range(n):
        period=rng.choice((30.,45.,60.))
        next_release=rng.uniform(2.,period*.65)
        total=rng.uniform(1.,5.)*(2.3 if regime=='pl_pressure' else 1.)
        large=Mode('large',tuple(total/4 for _ in range(4)))
        small=Mode('small',(total*.22,total*.22))
        # Synthetic capability assumptions deliberately nonuniform; no monotone
        # neural controller quality is inferred from size or from this generator.
        qualified=rng.random()<.65
        options=(Option(period,large),Option(max(15.,period*.65),large),
                 Option(min(80.,period*1.35),large),Option(period,small,qualified))
        if full_grid:
            options+=tuple(Option(o.period,small,qualified) for o in options[1:3])
        deadline=next_release+rng.uniform(5.,24.)
        loops.append(Loop(str(i),next_release-period,next_release,deadline,
                          rng.uniform(65.,110.),options,rng.random()<.4))
    carry=() if rng.random()<.5 else (Job('carry','previous',0,18,(2.,2.),'immutable'),)
    return Scenario(tuple(loops),p,horizon=85.,carry=carry,initial_policy=('fifo',2))


def summarize(rows):
    out={}
    for method in rows[0]['methods']:
        entries=[r['methods'][method] for r in rows]
        comparable=[r for r in rows if r.get('oracle',{}).get('accepted')]
        bad_base=[r for r in comparable if not r['methods']['fixed']['accepted']]
        wall=[r['wall_ms'] for r in entries]
        out[method]=dict(total=len(entries),accepted=sum(r['accepted'] for r in entries),
            accepted_with_observed_delay=sum(r.get('accepted_with_observed_delay',r['accepted']) for r in entries),
            oracle_feasible=len(comparable),
            recovered_oracle_feasible_bad_base=sum(r['methods'][method]['accepted'] for r in bad_base),
            oracle_feasible_bad_base=len(bad_base),
            wall_ms=dict(mean=float(np.mean(wall)),p99=float(np.percentile(wall,99)),max=max(wall)),
            checks_mean=float(np.mean([r['checks'] for r in entries])),checks_max=max(r['checks'] for r in entries))
    return out


def run(output, seeds, seed_start, initial_order, engine):
    output.mkdir(parents=True,exist_ok=False)
    predeclared=dict(seeds=seeds,seed_start=seed_start,initial_order=initial_order,engine=engine,
        regimes=['balanced','ps_pressure','pl_pressure','switch_pressure'],
        exact_oracle_loops=4,scale_loops=[6,12],horizon_ms=85,max_checks=5,max_edits=8,
        oracle_space='3 periods x 2 modes, qualified subsets only, x FIFO/EDF x engine legal boundaries; segmented also permits whole-job',
        assumptions='synthetic controller qualification, PS/PL timing and hold horizons; no board measurement',
        decision_reserve_ms=.5,unit='ms',acceptance='all modeled jobs meet end-to-end deadlines; drain after any miss')
    (output/'protocol.json').write_text(json.dumps(predeclared,indent=2),encoding='utf-8')
    rows=[]
    methods={'joint':{},'rate_only':dict(allowed={'rate'}),'mode_only':dict(allowed={'mode'}),
             'schedule_only':dict(allowed={'schedule'}),'demand_blind':dict(guided=False)}
    for n in (4,6,12):
        for regime in predeclared['regimes']:
            for seed in range(seeds):
                groups=(1,2,4) if engine=='segmented' else (4,)
                scenario=replace(random_case(seed_start+seed,n,regime,full_grid=True),
                                 initial_policy=(initial_order,groups[-1]),legal_groups=groups)
                row=dict(seed=seed_start+seed,n=n,regime=regime,scenario=asdict(scenario),methods={})
                start=time.perf_counter_ns()
                baseline=simulate(scenario,(0,)*n,scenario.initial_policy)
                row['methods']['fixed']=dict(accepted=baseline['feasible'],checks=1,
                    wall_ms=(time.perf_counter_ns()-start)/1e6,misses=len(baseline['misses']))
                for name,kwargs in methods.items():
                    start=time.perf_counter_ns();cpu=time.process_time_ns()
                    result=heuristic(scenario,**kwargs)
                    wall=(time.perf_counter_ns()-start)/1e6;elapsed_cpu=(time.process_time_ns()-cpu)/1e6
                    row['methods'][name]={k:v for k,v in result.items() if k!='forecast'}
                    row['methods'][name].update(wall_ms=wall,cpu_ms=elapsed_cpu,
                        misses=len(result['forecast']['misses']))
                    # Observed Python decision duration may exceed the assumed
                    # reserve. Recheck the returned choice with that delay. This
                    # is retrospective sensitivity, not certified online WCET.
                    charged=replace(scenario,platform=replace(scenario.platform,
                        decision=max(scenario.platform.decision,wall)))
                    row['methods'][name]['accepted_with_observed_delay'] = result['accepted'] and simulate(
                        charged,result['choice'],result['policy'])['feasible']
                    # An accepted result is independently rerun with its exact choice.
                    if result['accepted']:
                        assert simulate(scenario,result['choice'],result['policy'])['feasible']
                if n==4:
                    start=time.perf_counter_ns();row['oracle']=exhaustive(scenario)
                    row['oracle']['wall_ms']=(time.perf_counter_ns()-start)/1e6
                    assert row['oracle']['accepted'] or not any(r['accepted'] for r in row['methods'].values())
                rows.append(row)
            print('Completed',n,regime,'cases',len(rows),flush=True)
    summary=dict(protocol=predeclared,by_size={str(n):summarize([r for r in rows if r['n']==n]) for n in (4,6,12)},
        by_regime={regime:summarize([r for r in rows if r['n']==4 and r['regime']==regime]) for regime in predeclared['regimes']},
        false_acceptances_vs_oracle=0,
        interpretation='No physical flight or FPGA verified. Not a FALCON comparison. Heuristic misses retained.',
        python=platform.python_version(),os=platform.platform(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('bottleneck_joint.py')]})
    (output/'cases.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['by_size']['4'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=12)
    p.add_argument('--seed-start',type=int,default=20000)
    p.add_argument('--initial-order',choices=('fifo','edf'),default='edf')
    p.add_argument('--engine',choices=('segmented','whole'),default='segmented')
    a=p.parse_args();run(a.output,a.seeds,a.seed_start,a.initial_order,a.engine)
