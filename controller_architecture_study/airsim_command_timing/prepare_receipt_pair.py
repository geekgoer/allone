"""Prepare equal-probe ACK/receipt policy arms from the frozen Route-B sources."""

import argparse
import json
from pathlib import Path
import shutil

from .prepare_ps_trace import edit, prepare


def build(source, config, output):
    prepare(source, output / 'source')
    code = output / 'source'
    shutil.copyfile(Path(__file__).with_name('receipt_policy.hpp'), code / 'receipt_policy.hpp')
    edit(code / 'rpc/airsim_rpc.hpp', [
        ('    bool ping_ok = false;', '    bool ping_ok = false;\n    std::vector<std::uint64_t> receipt;'),
    ])
    edit(code / 'rpc/airsim_rpc.cpp', [
        ('    p.ping_ok = ping();', '''    if (trace_id != 0) {
        p.receipt = impl_->rpc.call("falconTracePing", trace_id).as<std::vector<std::uint64_t>>();
        p.ping_ok = p.receipt.size() == 4;
    } else {
        p.ping_ok = ping();
    }'''),
    ])
    edit(code / 'adaptive_runtime.cpp', [
        ('#include "adaptive_live.hpp"', '#include "adaptive_live.hpp"\n#include "receipt_policy.hpp"'),
        ('        writej(out+"/trace_clock_before.json",trace_clock_samples(*manager));',
         '        const auto clock_before=trace_clock_samples(*manager);\n'
         '        writej(out+"/trace_clock_before.json",clock_before);\n'
         '        std::vector<receipt_policy::ClockSample> receipt_calibration;\n'
         '        for(const auto& sample:clock_before)receipt_calibration.push_back({{sample.at(0).get<std::int64_t>(),\n'
         '            sample.at(1).get<std::int64_t>(),sample.at(2).get<std::int64_t>()}});\n'
         '        const bool use_adoption_receipt=config.value("use_adoption_receipt",false);'),
        ('                const double post_sample=double(ack_us-output_ready)/1000.;\n'
         '                const double reserve_sample=double(ack_us-row.at("control_converted_us").get<std::uint64_t>())/1000.;\n'
         '                if(command_retime&&row["status"]=="command_return_late"&&reserve_sample>row["period_ms"].get<double>()){',
         '''                const auto converted=row.at("control_converted_us").get<std::int64_t>()*1000;
                const auto deadline=std::int64_t(std::ceil(admission.deadline_ms*1e6));
                const auto receipt=receipt_policy::determine(use_adoption_receipt,receipt_calibration,
                    {ack.ping_start_ns, ack.receipt.empty()?0:std::int64_t(ack.receipt[0]), ack.ping_return_ns},
                    trace_id,ack.receipt,ack.submit_ns,std::int64_t(ack_us)*1000,deadline,
                    converted+std::int64_t(row["period_ms"].get<double>()*1e6));
                const double post_sample=double(receipt.cost_end_ns-std::int64_t(output_ready)*1000)/1e6;
                const double reserve_sample=double(receipt.cost_end_ns-converted)/1e6;
                row["receipt_policy"]={{"evidence",receipt.evidence},{"enabled",use_adoption_receipt},
                    {"trigger",receipt.trigger},{"adoption_lower_ns",receipt.adoption_lower_ns},
                    {"adoption_upper_ns",receipt.adoption_upper_ns},{"budget_end_ns",receipt.cost_end_ns}};
                if(command_retime&&receipt.trigger){'''),
        ('"native command+ping return proxy, not physical actuator timestamp"',
         '"legacy ACK status and hold use RPC return; receipt arm changes only retime trigger and local command cost"'),
    ])
    cfg = json.loads(config.read_text(encoding='utf-8'))
    if not (cfg['isolate_command_costs'] and cfg['single_command_check'] and cfg['enable_command_retime']):
        raise ValueError('Use the frozen retime comparison config')
    for arm, switch in (('ping', False), ('receipt', True)):
        (output / f'config_{arm}.json').write_text(json.dumps(dict(cfg, use_adoption_receipt=switch), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.config, args.output)


if __name__ == '__main__':
    main()
