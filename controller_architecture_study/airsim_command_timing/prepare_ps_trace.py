"""Generate a diagnostic copy of the frozen PS controller, retaining the ping policy."""

import argparse
from pathlib import Path
import shutil

from .prepare_blocks import replace_once


def edit(path, replacements):
    text = path.read_text(encoding='utf-8')
    for old, new in replacements:
        text = replace_once(text, old, new)
    path.write_text(text, encoding='utf-8', newline='\n')


def prepare(source, output):
    if source.resolve() in output.resolve().parents:
        raise ValueError('Output must be outside the frozen source tree')
    shutil.copytree(source, output)
    edit(output / 'rpc/airsim_rpc.hpp', [
        ('float duration);', 'float duration, std::uint64_t trace_id = 0);\n    std::uint64_t trace_clock();'),
    ])
    edit(output / 'rpc/airsim_rpc.cpp', [
        ('float duration) {', 'float duration, std::uint64_t trace_id) {'),
        ('    impl_->last_move = impl_->rpc.async_call("moveByAngleRatesThrottle", wx, wy,\n'
         '                                            wz, throttle, duration, vehicle);',
         '    if (trace_id != 0)\n'
         '        impl_->last_move = impl_->rpc.async_call("falconTraceMove", trace_id, wx, wy, wz,\n'
         '                                                throttle, duration, 0u, vehicle);\n'
         '    else\n'
         '        impl_->last_move = impl_->rpc.async_call("moveByAngleRatesThrottle", wx, wy,\n'
         '                                                wz, throttle, duration, vehicle);'),
        ('void Client::pause', 'std::uint64_t Client::trace_clock() {\n'
         '    return impl_->rpc.call("falconTraceClock").as<std::uint64_t>();\n}\n'
         'void Client::pause'),
    ])
    edit(output / 'adaptive_runtime.cpp', [
        ('int main(int argc,char** argv){', '''Json trace_clock_samples(psairsim::Client& client) {
    Json samples=Json::array();
    for(int i=0;i<32;++i){
        const auto before=psairsim::monotonic_ns();
        const auto server=client.trace_clock();
        const auto after=psairsim::monotonic_ns();
        samples.push_back({before,server,after});
    }
    return samples;
}

int main(int argc,char** argv){'''),
        ('    double duration=20.;', '    int port=41591; std::atomic<std::uint64_t> trace_next{0};\n'
         '    std::uint64_t trace_base=0;\n    double duration=20.;'),
        ('            else if(option=="--duration-s")',
         '            else if(option=="--port")port=std::stoi(value);\n'
         '            else if(option=="--trace-base")trace_base=std::stoull(value);\n'
         '            else if(option=="--duration-s")'),
        ('        if(enable_skip&&policy',
         '        if(port<=1023||port>65535||port==41451||trace_base==0||trace_base%32768!=0||\n'
         '           trace_base>=(1ULL<<62)||duration>40.)\n'
         '            throw std::runtime_error("Diagnostic port and aligned trace base required; use at most 40s");\n'
         '        if(enable_skip&&policy'),
        ('        manager.reset(new psairsim::Client(host,41451,2000));',
         '        if(setup.value("host",std::string())!=host||setup.value("port",0)!=port)\n'
         '            throw std::runtime_error("Prepared scene belongs to a different endpoint");\n'
         '        manager.reset(new psairsim::Client(host,static_cast<std::uint16_t>(port),2000));'),
        ('        psdpu::LocalDpu dpu;', '        writej(out+"/trace_clock_before.json",trace_clock_samples(*manager));\n'
         '        psdpu::LocalDpu dpu;'),
        ('new psairsim::Client(host,41451,2000));const auto raw=',
         'new psairsim::Client(host,static_cast<std::uint16_t>(port),2000));const auto raw='),
        ('                const auto ack=v->client->move_rates_throttle(v->name,wire.roll_rate,wire.pitch_rate,wire.yaw_rate,wire.throttle,wire.duration);',
         '                const auto trace_index=trace_next.fetch_add(1,std::memory_order_relaxed);\n'
         '                if(trace_index>=32768)throw std::runtime_error("Diagnostic trace capacity exceeded");\n'
         '                const auto trace_id=trace_base+trace_index;row["command_trace_id"]=trace_id;\n'
         '                const auto ack=v->client->move_rates_throttle(v->name,wire.roll_rate,wire.pitch_rate,wire.yaw_rate,wire.throttle,wire.duration,trace_id);'),
        ('        // Drain events already latched',
         '        writej(out+"/trace_clock_after.json",trace_clock_samples(*manager));\n'
         '        // Drain events already latched'),
    ])
    edit(output / 'build_control.sh', [
        ('    "$original/build_adaptive_005/dependencies/libpsairsim.a" "$original/build_adaptive_005/dependencies/librpc.a"',
         '    "$study/rpc/airsim_rpc.cpp" -I"$RPC_HEADERS" "$original/build_adaptive_005/dependencies/librpc.a"'),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
