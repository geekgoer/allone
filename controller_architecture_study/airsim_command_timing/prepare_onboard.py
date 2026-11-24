"""Build a dedicated two-vehicle AirSim experiment with an in-firmware-path pilot."""

import argparse
import json
from pathlib import Path
import shutil

from .prepare_blocks import prepare, replace_once


def build(blocks, output, settings, port):
    prepare(blocks, output, settings, port)
    plugin = output / 'Plugins/AirSim/Source'
    lib = plugin / 'AirLib/include'
    shutil.copyfile(Path(__file__).with_name('onboard_pilot.hpp'), lib / 'common/onboard_pilot.hpp')
    api = lib / 'vehicles/multirotor/firmwares/simple_flight/SimpleFlightApi.hpp'
    text = api.read_text(encoding='utf-8')
    text = replace_once(text, '#include "common/tagged_trace.hpp"',
                        '#include "common/tagged_trace.hpp"\n#include "common/onboard_pilot.hpp"')
    text = replace_once(text, ': vehicle_params_(vehicle_params)',
                        ': pilot_vehicle_(vehicle_setting->vehicle_name), vehicle_params_(vehicle_params)')
    text = replace_once(text, '            if (trace_pending_ == 0) {', '''            if (falcon_pilot::active(pilot_vehicle_) && firmware_->offboardApi().hasApiControl()) {
                const auto position = getKinematicsEstimated().pose.position;
                const falcon_pilot::Position pos{{position.x(), position.y(), position.z()}};
                auto next = falcon_pilot::choose(pilot_vehicle_, pos, falcon_trace::now_ns());
                if (next.id) {
                    using Kind = simple_flight::GoalModeType;
                    const simple_flight::GoalMode mode(Kind::VelocityWorld, Kind::VelocityWorld,
                                                       Kind::AngleLevel, Kind::PositionWorld);
                    const simple_flight::Axis4r goal(float(next.vy), float(next.vx), 0.f, float(next.z));
                    std::string message;
                    if (!firmware_->offboardApi().setGoalAndMode(&goal, &mode, message)) next = {};
                }
                firmware_->update();
                falcon_pilot::record(pilot_vehicle_, pos, falcon_trace::now_ns(), next);
                return;
            }
            if (trace_pending_ == 0) {''')
    text = replace_once(text, '        const MultiRotorParams* vehicle_params_;',
                        '        const std::string pilot_vehicle_;\n        const MultiRotorParams* vehicle_params_;')
    api.write_text(text, encoding='utf-8')
    server = plugin / 'Vehicles/Multirotor/tagged_multirotor_server.hpp'
    text = server.read_text(encoding='utf-8')
    text = replace_once(text, '#include "common/tagged_trace.hpp"',
                        '#include "common/tagged_trace.hpp"\n#include "common/onboard_pilot.hpp"')
    text = replace_once(text, '        rpc_server->bind("falconTraceClock", [] { return now_ns(); });',
                        '''        rpc_server->bind("falconTraceClock", [] { return now_ns(); });
        rpc_server->bind("falconPilotStart", falcon_pilot::start);
        rpc_server->bind("falconPilotSubmit", falcon_pilot::submit);
        rpc_server->bind("falconPilotStop", falcon_pilot::stop);''')
    server.write_text(text, encoding='utf-8')
    config = output / 'diagnostic_settings.json'
    data = json.loads(config.read_text(encoding='utf-8'))
    data['Vehicles'] = {v: data['Vehicles'][v] for v in ('Drone1', 'Drone2')}
    config.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blocks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--port', type=int, default=41592)
    args = parser.parse_args()
    build(args.blocks, args.output, args.settings, args.port)


if __name__ == '__main__':
    main()
