"""Freeze an independent adoption stream comparison on the ten-UAV PS controller."""

import argparse
import json
from pathlib import Path
import shutil

from .prepare_ps_trace import edit, prepare


def build(source, config, output, pin_release_pair=False):
    code = output / 'source'
    prepare(source, code)
    for name in ('receipt_policy.hpp', 'adoption_inbox.hpp'):
        shutil.copyfile(Path(__file__).with_name(name), code / name)
    edit(code / 'rpc/airsim_rpc.hpp', [
        ('    std::uint64_t trace_clock();', '    std::uint64_t trace_clock();\n'
         '    std::uint64_t adoption_cursor();\n'
         '    std::vector<std::vector<std::uint64_t>> adoption_events(std::uint64_t cursor);'),
    ])
    edit(code / 'rpc/airsim_rpc.cpp', [
        ('void Client::pause', '''std::uint64_t Client::adoption_cursor() {
    return impl_->rpc.call("falconAdoptionCursor").as<std::uint64_t>();
}
std::vector<std::vector<std::uint64_t>> Client::adoption_events(std::uint64_t cursor) {
    return impl_->rpc.call("falconAdoptionEvents", cursor).as<std::vector<std::vector<std::uint64_t>>>();
}
void Client::pause'''),
    ])
    edit(code / 'adaptive_runtime.cpp', [
        ('#include "adaptive_live.hpp"', '#include "adaptive_live.hpp"\n#include "adoption_inbox.hpp"'),
        ('        writej(out+"/trace_clock_before.json",trace_clock_samples(*manager));', '''        const auto clock_before=trace_clock_samples(*manager);
        writej(out+"/trace_clock_before.json",clock_before);
        std::vector<receipt_policy::ClockSample> receipt_calibration;
        for(const auto& sample:clock_before)receipt_calibration.push_back({{sample.at(0).get<std::int64_t>(),
            sample.at(1).get<std::int64_t>(),sample.at(2).get<std::int64_t>()}});
        const bool use_adoption_receipt=config.at("use_adoption_event").get<bool>();
        adoption_events::Inbox adoption(host,static_cast<std::uint16_t>(port),trace_base);'''),
        ('            std::uint64_t late_version=0,late_cycle=0;', '            std::uint64_t late_version=0,late_cycle=0;\n'
         '            adoption_events::Observation pending_adoption;'),
        ('                const auto release_row=calendar.row(v->name,slot->cycle);', '''                if(pending_adoption.id){
                    const auto previous=pending_adoption;
                    pending_adoption={};
                    const auto event=adoption.lookup(previous.id);
                    const auto resolution=adoption_events::resolve(use_adoption_receipt,receipt_calibration,previous,event);
                    if(previous.version!=v->version)throw std::runtime_error("Adoption version changed within worker");
                    calendar.update_command_costs(v->name,double(resolution.cost_end_ns-previous.output)/1e6,
                                                  double(resolution.cost_end_ns-previous.converted)/1e6);
                    if(command_retime&&resolution.trigger){late_version=previous.version;late_cycle=previous.cycle;}
                    row["prior_adoption_decision"]={{"cycle",previous.cycle},{"id",previous.id},
                        {"evidence",resolution.evidence},{"event_received_ns",event.received_ns},
                        {"old_trigger",previous.ack>=previous.deadline&&previous.ack>previous.reserve_limit},
                        {"new_trigger",resolution.trigger},{"old_reserve_ms",double(previous.ack-previous.converted)/1e6},
                        {"used_reserve_ms",double(resolution.cost_end_ns-previous.converted)/1e6}};
                }
                const auto release_row=calendar.row(v->name,slot->cycle);'''),
        ('''                const double post_sample=double(ack_us-output_ready)/1000.;
                const double reserve_sample=double(ack_us-row.at("control_converted_us").get<std::uint64_t>())/1000.;
                if(command_retime&&row["status"]=="command_return_late"&&reserve_sample>row["period_ms"].get<double>()){
                    late_version=v->version;late_cycle=slot->cycle;
                }
                if(isolate_command_costs)calendar.update_command_costs(v->name,post_sample,reserve_sample);
                else costs.observe(0.,post_sample,0.,reserve_sample);''', '''                const auto converted=row.at("control_converted_us").get<std::int64_t>()*1000;
                pending_adoption={trace_id,v->version,slot->cycle,ack.submit_ns,converted,
                    std::int64_t(output_ready)*1000,std::int64_t(ack_us)*1000,
                    std::int64_t(std::ceil(admission.deadline_ms*1e6)),
                    converted+std::int64_t(row["period_ms"].get<double>()*1e6)};'''),
        ('        writej(out+"/trace_clock_after.json",trace_clock_samples(*manager));', '''        writej(out+"/trace_clock_after.json",trace_clock_samples(*manager));
        adoption.close();
        Json events=Json::array();
        for(const auto& event:adoption.captured())events.push_back({event.id,event.start_ns,event.end_ns,event.received_ns});
        writej(out+"/trace_live_events.json",{{"failed",adoption.failed()},{"events",std::move(events)}});'''),
    ])
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if not (cfg['isolate_command_costs'] and cfg['single_command_check'] and cfg['enable_command_retime']):
        raise ValueError('Use the frozen retime comparison config')
    if pin_release_pair:
        maintained = Path(__file__).resolve().parents[1] / 'ps_planner_repair' / 'live'
        for name in ('scheduler.cpp', 'scheduler.hpp', 'live_calendar.cpp', 'live_calendar.hpp', 'live_calendar_tests.cpp'):
            shutil.copyfile(maintained / name, code / 'scheduler' / name)
        edit(code / 'adaptive_runtime.cpp', [
            ('const auto update=calendar.try_command_retime(v->name,slot->cycle,a::now_ms);',
             'Json retime_guard;\n                const auto update=calendar.try_command_retime(v->name,slot->cycle,a::now_ms,&retime_guard);'),
            ('{"wall_us",a::now_us()-start},{"accepted",!update.is_null()}};',
             '{"wall_us",a::now_us()-start},{"accepted",!update.is_null()},{"guard",std::move(retime_guard)}};'),
        ])
        arms = (('event_schedule', dict(use_adoption_event=True, retime_pin_release=False)),
                ('pin_release', dict(use_adoption_event=True, retime_pin_release=True)))
    else:
        arms = (('event_log', dict(use_adoption_event=False)), ('event_schedule', dict(use_adoption_event=True)))
    for name, changes in arms:
        (output / f'config_{name}.json').write_text(json.dumps(dict(cfg, **changes), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pin-release-pair', action='store_true')
    args = parser.parse_args()
    build(args.source, args.config, args.output, args.pin_release_pair)


if __name__ == '__main__':
    main()
