#pragma once

#include "common/Common.hpp"
#include "vehicles/multirotor/api/MultirotorRpcLibServer.hpp"
#include "common/tagged_trace.hpp"

STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif
#include "common/common_utils/MinWinDefines.hpp"
#undef NOUSER
#include "common/common_utils/WindowsApisCommonPre.hpp"
#undef FLOAT
#pragma push_macro("check")
#undef check
#include "rpc/server.h"
#pragma pop_macro("check")
#include "common/common_utils/WindowsApisCommonPost.hpp"
STRICT_MODE_ON

#include <cmath>
#include <thread>

namespace falcon_trace {

class TaggedMultirotorServer : public msr::airlib::MultirotorRpcLibServer {
public:
    TaggedMultirotorServer(msr::airlib::ApiProvider* provider, const std::string& address, std::uint16_t port)
        : MultirotorRpcLibServer(provider, address, port) {
        auto* rpc_server = static_cast<rpc::server*>(getServer());
        rpc_server->bind("falconTraceClock", [] { return now_ns(); });
        rpc_server->bind("falconTracePing", [](Stamp id) { return snapshot(id); });
        rpc_server->bind("falconTraceRead", [](const std::vector<Stamp>& ids) { return read(ids); });
        rpc_server->bind("falconTraceMove", [this](Stamp id, float roll, float pitch, float yaw,
                                                   float throttle, float duration, unsigned delay_ms,
                                                   const std::string& vehicle) {
            if (!std::isfinite(roll) || !std::isfinite(pitch) || !std::isfinite(yaw) ||
                !std::isfinite(throttle) || throttle < 0 || throttle > 1 ||
                !std::isfinite(duration) || duration <= 0 || duration > 1.0f || delay_ms > 80)
                throw std::invalid_argument("Invalid diagnostic command");
            Command command(id);
            if (delay_ms != 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            return getVehicleApi(vehicle)->moveByAngleRatesThrottle(roll, pitch, yaw, throttle, duration);
        });
    }
};

} // namespace falcon_trace
