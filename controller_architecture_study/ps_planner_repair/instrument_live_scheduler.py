"""Add non-overlapping CPU/wall stages to the independently built PS planner."""
from pathlib import Path


def instrument(source: str) -> str:
    source = source.replace('#include "scheduler.hpp"',
        '#include "scheduler.hpp"\n#include "profile.hpp"\nnamespace ps_profile {thread_local State state;}', 1)
    replacements = [
        ('Modes parse_modes(const Json& raw) {', 'Modes parse_modes(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Choice parse_choice(const Json& raw) {', 'Choice parse_choice(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Jobs parse_jobs(const Json& raw) {', 'Jobs parse_jobs(const Json& raw) { PS_SCOPE(Preprocess);'),
        ('Json simulate(const std::vector<Demand>& jobs,const Sequence& order,double decision,double block) {',
         'Json simulate(const std::vector<Demand>& jobs,const Sequence& order,double decision,double block) { PS_SCOPE(Simulation);'),
        ('    Json slack=Json::object(),misses=', '    PS_SCOPE(SimulationReport);\n    Json slack=Json::object(),misses='),
        ('double tail(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){',
         'double tail(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);'),
        ('double dispatch(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){',
         'double dispatch(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now){ PS_SCOPE(Budget);'),
        ('Json evaluate_impl(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now,double decision,double block){',
         'Json evaluate_impl(const Jobs&jobs,const Sequence&order,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(GlobalCheck);'),
        ('Json diagnose_impl(const Jobs&jobs,const Choice&choice,const Modes&modes,double now,double decision,double block){',
         'Json diagnose_impl(const Jobs&jobs,const Choice&choice,const Modes&modes,double now,double decision,double block){ PS_SCOPE(Diagnosis);'),
        ('void validate(const Json&calendar,const Json&raw_modes){',
         'void validate(const Json&calendar,const Json&raw_modes){ PS_SCOPE(Preprocess);'),
        ('Json plan(const Json&calendar,const Json&raw_modes,const Json&options){',
         'Json plan(const Json&calendar,const Json&raw_modes,const Json&options){ PS_SCOPE(Other);'),
        ('        if(candidates>=max_candidates||checks>=max_checks)return;',
         '        PS_SCOPE(Candidate);\n        if(candidates>=max_candidates||checks>=max_checks)return;'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Planner source changed at ' + before)
        source = source.replace(before, after, 1)
    return source


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    a.output.write_text(instrument(a.source.read_text(encoding='utf-8')), encoding='utf-8')
