"""Copy Blocks and instrument SimpleFlight in the copy only."""

import argparse
import json
from pathlib import Path
import shutil


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected AirSim source around {old[:100]!r}")
    return text.replace(old, new, 1)


def instrument_api(text):
    text = replace_once(text, '#include "firmware/Firmware.hpp"',
                        '#include "firmware/Firmware.hpp"\n#include "common/tagged_trace.hpp"')
    calls = 'firmware_->offboardApi().setGoalAndMode(&goal, &mode, message);'
    if calls not in text:
        raise ValueError("Missing SimpleFlight goal setters")
    text = text.replace(calls, 'setProbeGoal(goal, mode, message);')
    start = text.index('virtual void commandAngleRatesThrottle(')
    end = text.index('\n        }', start)
    text = text[:start] + replace_once(text[start:end], 'setProbeGoal(goal, mode, message);',
                                       'setProbeGoal(goal, mode, message, falcon_trace::current_id());') + text[end:]
    for signature in ('virtual void resetImplementation() override',
                      'virtual void enableApiControl(bool is_enabled) override',
                      'virtual bool armDisarm(bool arm) override'):
        text = replace_once(text, signature + '\n        {', signature + '\n        {\n'
                            '            std::lock_guard<std::mutex> lock(trace_goal_mutex_);\n'
                            '            trace_pending_ = 0;')
    text = replace_once(text, '            firmware_->update();', '''            std::lock_guard<std::mutex> lock(trace_goal_mutex_);
            if (trace_pending_ == 0) {
                firmware_->update();
                return;
            }
            const auto started = falcon_trace::now_ns();
            firmware_->update();
            const auto ended = falcon_trace::now_ns();
            auto& offboard = firmware_->offboardApi();
            if (offboard.hasApiControl() && offboard.getGoalValue().equals4(trace_goal_) &&
                offboard.getGoalMode().equals4(trace_mode_))
                falcon_trace::mark_firmware(trace_pending_, started, ended);
            trace_pending_ = 0;''')
    return replace_once(text, '        const MultiRotorParams* vehicle_params_;', '''        void setProbeGoal(const simple_flight::Axis4r& goal, const simple_flight::GoalMode& mode,
                          std::string& message, falcon_trace::Stamp id = 0)
        {
            std::lock_guard<std::mutex> lock(trace_goal_mutex_);
            if (!firmware_->offboardApi().setGoalAndMode(&goal, &mode, message))
                return;
            trace_pending_ = id;
            if (id != 0) {
                trace_goal_ = goal;
                trace_mode_ = mode;
                falcon_trace::mark_goal(id);
            }
        }

        std::mutex trace_goal_mutex_;
        falcon_trace::Stamp trace_pending_ = 0;
        simple_flight::Axis4r trace_goal_;
        simple_flight::GoalMode trace_mode_ = simple_flight::GoalMode::getUnknown();
        const MultiRotorParams* vehicle_params_;''')


def prepare(source, output, settings_file, port):
    source, output = source.resolve(), output.resolve()
    if source in output.parents or output == source:
        raise ValueError("Put the diagnostic copy outside the original Blocks directory")
    if not (source / 'Blocks.uproject').is_file() or port == 41451 or not 1024 <= port <= 65535:
        raise ValueError("Expected a built Blocks project and a dedicated diagnostic port")
    output.mkdir(parents=True)
    for name in ('Source', 'Content', 'Config', 'Plugins/AirSim/Source', 'Plugins/AirSim/Content'):
        shutil.copytree(source / name, output / name)
    for name in ('Blocks.uproject', 'Plugins/AirSim/AirSim.uplugin'):
        shutil.copyfile(source / name, output / name)

    assets = Path(__file__).parent
    plugin = output / 'Plugins/AirSim/Source'
    include = plugin / 'AirLib/include'
    shutil.copyfile(assets / 'tagged_trace.hpp', include / 'common/tagged_trace.hpp')
    vehicle = plugin / 'Vehicles/Multirotor'
    shutil.copyfile(assets / 'tagged_multirotor_server.hpp', vehicle / 'tagged_multirotor_server.hpp')
    api = include / 'vehicles/multirotor/firmwares/simple_flight/SimpleFlightApi.hpp'
    api.write_text(instrument_api(api.read_text(encoding='utf-8')), encoding='utf-8')
    server = vehicle / 'SimModeWorldMultiRotor.cpp'
    text = server.read_text(encoding='utf-8')
    text = replace_once(text, '#include "common/SteppableClock.hpp"',
                        '#include "common/SteppableClock.hpp"\n#ifndef AIRLIB_NO_RPC\n'
                        '#include "tagged_multirotor_server.hpp"\n#endif')
    text = replace_once(text, 'new msr::airlib::MultirotorRpcLibServer(', 'new falcon_trace::TaggedMultirotorServer(')
    server.write_text(text, encoding='utf-8')

    settings = json.loads(settings_file.read_text(encoding='utf-8-sig'))
    settings.update(ApiServerPort=port, LocalHostIp='127.0.0.1')
    (output / 'diagnostic_settings.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--port', type=int, default=41591)
    args = parser.parse_args()
    prepare(args.blocks, args.output, args.settings, args.port)


if __name__ == '__main__':
    main()
