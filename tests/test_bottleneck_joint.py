import itertools
import random
import unittest
from dataclasses import replace

from controller_architecture_study.bottleneck_joint import (
    Job, Loop, Mode, Option, Platform, Scenario, exhaustive, heuristic, make_jobs,
    simulate, window_demand, witness, blocking_contributions)


class BottleneckJointTests(unittest.TestCase):
    def setUp(self):
        self.mode=Mode('fixed',(2.,2.))
        self.p=Platform(prep=0,dispatch=0,post=0,ingress=0,tail=0,switch=0,decision=0)

    def loop(self, name='a', **kw):
        d=dict(name=name,previous_release=-20.,next_release=0.,deadline=10.,hold_bound=80.,
               options=(Option(20.,self.mode),),release_locked=True)
        return Loop(**(d|kw))

    def test_isolated_response_and_ps_pl_overlap(self):
        s=Scenario((self.loop(),),self.p,horizon=1)
        r=simulate(s,(0,),('edf',1))
        self.assertEqual(r['finish']['a:0'],4.)
        self.assertTrue(r['feasible'])
        p=Platform(prep=2,dispatch=0,post=0,ingress=0,tail=0,switch=0,decision=0)
        r=simulate(Scenario((self.loop(),self.loop('b')),p,horizon=1),(0,0),('edf',1))
        self.assertTrue(any(a['resource']=='ps' and b['resource']=='pl' and
            min(a['end'],b['end'])>max(a['start'],b['start']) for a in r['timeline'] for b in r['timeline']))

    def test_segment_is_nonpreemptible(self):
        a=self.loop(deadline=20,options=(Option(30,Mode('long',(8.,))),))
        b=self.loop('b',next_release=1,deadline=4,options=(Option(30,Mode('short',(1.,))),))
        r=simulate(Scenario((a,b),self.p,horizon=2),(0,0),('edf',1))
        self.assertEqual(r['finish']['b:0'],9.)
        self.assertFalse(r['feasible'])

    def test_native_front_wait_overlaps_but_stays_on_each_response_path(self):
        p = replace(self.p, front_wait=8., prep=2., decision=4., rearm=1.)
        options = (Option(20., Mode('short', (1.,))),)
        a = self.loop(deadline=14., options=options)
        b = self.loop('b', deadline=14., options=options)
        s = Scenario((a, b), p, horizon=1)
        r = simulate(s, (0, 0), ('edf', 1))
        self.assertTrue(r['feasible'])
        self.assertEqual(r['command_finish'], {'a:0': 12., 'b:0': 14.})
        self.assertEqual(r['finish']['b:0'], 15.)
        self.assertEqual(window_demand(r['jobs'], p, dict(start=0., end=14., kind='ps')), 4.)
        s = replace(s, loops=(replace(a, deadline=10.),))
        r = simulate(s, (0,), ('edf', 1))
        self.assertFalse(r['feasible'])
        self.assertEqual(witness(s, r['jobs'])['deficit'], 1.)

    def test_response_must_fit_period_even_if_physical_deadline_wide(self):
        l=self.loop(deadline=100,options=(Option(3,self.mode),))
        r=simulate(Scenario((l,),self.p,horizon=1),(0,),('edf',1))
        self.assertFalse(r['feasible'])

    def test_locked_release_does_not_move(self):
        l=self.loop(next_release=18,options=(Option(55,self.mode),Option(20,self.mode)))
        s=Scenario((l,),self.p,horizon=19)
        self.assertEqual(make_jobs(s,(1,),1)[0].release,18)

    def test_model_only_change_keeps_committed_release_phase(self):
        options = (Option(20., self.mode), Option(20., Mode('fast', (1.,))))
        loop = self.loop(previous_release=-15., next_release=9., options=options, release_locked=False)
        scenario = Scenario((loop,), self.p, horizon=1.)
        old, = make_jobs(scenario, (0,), 1)
        new, = make_jobs(scenario, (1,), 1)
        self.assertEqual((new.release, new.deadline), (old.release, old.deadline))

    def test_future_rate_change_cannot_erase_queued_demand(self):
        carry=Job('old','a',0,8,(6.,),'old_model')
        l=self.loop(next_release=20,options=(Option(20,self.mode),Option(80,self.mode)))
        s=Scenario((l,),self.p,horizon=22,carry=(carry,))
        interval=dict(start=0,end=8,kind='pl')
        self.assertEqual(window_demand(make_jobs(s,(0,),1),self.p,interval),6)
        self.assertEqual(window_demand(make_jobs(s,(1,),1),self.p,interval),6)

    def test_unknown_controller_never_becomes_qualified(self):
        l=self.loop(options=(Option(20,self.mode),Option(20,Mode('fast',(.1,)),False)))
        with self.assertRaises(ValueError):make_jobs(Scenario((l,),self.p),(1,),1)
        self.assertEqual(exhaustive(Scenario((l,),self.p,horizon=1))['configurations'],4)

    def test_controller_bound_not_restarted_for_first_obligation(self):
        l=self.loop(next_release=6,deadline=7)
        self.assertFalse(simulate(Scenario((l,),self.p,horizon=7),(0,),('edf',1))['feasible'])

    def test_forecast_window_cannot_hide_current_obligation(self):
        l=self.loop(next_release=20,deadline=7)
        result=simulate(Scenario((l,),self.p,horizon=10),(0,),('edf',1))
        self.assertEqual(len(result['jobs']),1)
        self.assertFalse(result['feasible'])

    def test_demand_certificate_and_cross_boundary_omission(self):
        jobs=(Job('a','a',0,10,(6.,),'a'),Job('b','b',0,10,(6.,),'b'))
        s=Scenario((),self.p)
        w=witness(s,jobs)
        self.assertEqual((w['kind'],w['deficit']),('pl',2))
        self.assertEqual(window_demand(jobs,self.p,dict(start=2,end=10,kind='pl')),0)

    def test_necessary_pruning_against_unpruned_simulator(self):
        rng=random.Random(112)
        for _ in range(30):
            loops=tuple(self.loop(str(i),next_release=rng.uniform(0,3),deadline=rng.uniform(3,20),
                options=(Option(20,Mode('a',(rng.uniform(.1,4),)*2)),)) for i in range(2))
            s=Scenario(loops,self.p,horizon=4)
            for policy in itertools.product(('fifo','edf'),(1,2)):
                r=simulate(s,(0,0),policy)
                if r['feasible']:self.assertLessEqual(witness(s,r['jobs'])['deficit'],1e-8)

    def test_global_check_rejects_harming_another_loop(self):
        a=self.loop(deadline=8,options=(Option(20,Mode('a',(5.,))),))
        b=self.loop('b',deadline=4,options=(Option(20,Mode('b',(3.,))),))
        s=Scenario((a,b),self.p,horizon=1,initial_policy=('fifo',1))
        self.assertFalse(simulate(s,(0,0),('fifo',1))['feasible'])
        h=heuristic(s)
        self.assertTrue(h['accepted'])
        self.assertEqual(h['policy'][0],'edf')
        self.assertTrue(all(v>=-1e-8 for v in h['forecast']['slack'].values()))

    def test_bounded_checks_and_no_model_mutation_of_carry(self):
        s=Scenario((self.loop(deadline=.1),),self.p,horizon=1)
        h=heuristic(s,max_checks=2,max_edits=2)
        self.assertLessEqual(h['checks'],2)
        self.assertFalse(h['accepted'])

    def test_supported_boundary_matters_and_whole_job_dpu_cannot_preempt(self):
        a=self.loop(deadline=30,options=(Option(30,Mode('a',(4.,4.))),))
        b=self.loop('b',next_release=1,deadline=5,options=(Option(30,Mode('b',(1.,))),))
        s=Scenario((a,b),self.p,horizon=2,initial_policy=('edf',2))
        self.assertFalse(simulate(s,(0,0),('edf',2))['feasible'])
        self.assertTrue(simulate(s,(0,0),('edf',1))['feasible'])
        whole=Scenario((a,b),self.p,horizon=2,initial_policy=('edf',4),legal_groups=(4,))
        with self.assertRaises(ValueError):simulate(whole,(0,0),('edf',1))
        self.assertFalse(heuristic(whole)['accepted'])

    def test_blocking_credit_goes_to_actual_interferer(self):
        a=self.loop(deadline=30,options=(Option(30,Mode('a',(8.,))),))
        b=self.loop('b',next_release=1,deadline=5,options=(Option(30,Mode('b',(1.,))),))
        r=simulate(Scenario((a,b),self.p,horizon=2),(0,0),('edf',1))
        self.assertEqual(blocking_contributions(r,'b:0'),{('a:0','pl'):7.})

    def test_faster_rate_can_harm_other_loop(self):
        options=(Option(20,Mode('a',(6.,))),Option(10,Mode('a',(6.,))))
        a=self.loop(deadline=20,hold_bound=80,options=options)
        b=self.loop('b',next_release=5,deadline=25,hold_bound=80,options=(options[0],))
        s=Scenario((a,b),self.p,horizon=100)
        self.assertTrue(simulate(s,(0,0),('edf',1))['feasible'])
        # 0.6 + 0.3 = .9 remains feasible; use 8ms modes to cross capacity.
        a=self.loop(deadline=20,options=(Option(20,Mode('a',(8.,))),Option(10,Mode('a',(8.,)))))
        b=self.loop('b',next_release=5,deadline=25,options=(Option(20,Mode('b',(8.,))),))
        s=Scenario((a,b),self.p,horizon=100)
        self.assertTrue(simulate(s,(0,0),('edf',1))['feasible'])
        self.assertFalse(simulate(s,(1,0),('edf',1))['feasible'])


if __name__=='__main__':unittest.main()
