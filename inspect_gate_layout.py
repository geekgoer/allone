"""Inspect the generated three-drone gate layout without changing AirSim.

Default mode is completely offline: it only generates and prints the same
gate centers used by ``run_three_drones_gates.py``.  With ``--plot`` it sends
non-persistent debug points/lines to an already running AirSim instance; it
does not reset, arm, move vehicles, spawn actors, or destroy actors.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from environment import generate_gate_sequences
from reference_generator import gate_alignment_waypoint, gate_passage_waypoints


VEHICLES = ("Drone1", "Drone2", "Drone3")
COLORS = (
    [1.0, 0.1, 0.1, 1.0],
    [0.1, 1.0, 0.1, 1.0],
    [0.1, 0.4, 1.0, 1.0],
)


def _point(value: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(item): .3f}" for item in value) + ")"


def _print_layout(
    sequences: dict[str, list[object]],
    *,
    approach_distance: float,
    exit_distance: float,
) -> None:
    print("Gate centers and passage waypoints in vehicle-local AirSim NED (x, y, z):")
    print("  Gate frame: x is the passage direction; the visible opening is in y-z.")
    for vehicle in VEHICLES:
        print(f"\n{vehicle}")
        for gate in sequences[vehicle]:
            alignment = gate_alignment_waypoint([0.0, 0.0, -2.0], gate.center)
            before, after = gate_passage_waypoints(
                gate.center,
                gate.orientation,
                approach_distance=approach_distance,
                exit_distance=exit_distance,
            )
            print(
                f"  gate {gate.index}: center={_point(gate.center)} "
                f"align(example)={_point(alignment)} "
                f"before={_point(before)} after={_point(after)}"
            )


def _plot_layout(
    sequences: dict[str, list[object]],
    *,
    duration: float,
    point_size: float,
) -> None:
    from airsim_interface import require_airsim, vehicle_local_to_world_offset

    # AirSim's RPC layer expects a list of Vector3r objects here.  Passing
    # NumPy/Python lists reaches msgpack correctly but fails the server-side
    # type cast with ``RPCError: bad cast``.
    airsim = require_airsim()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    for vehicle, color in zip(VEHICLES, COLORS):
        world_offset = vehicle_local_to_world_offset(client, vehicle)
        centers = [
            airsim.Vector3r(
                float(gate.center[0] + world_offset[0]),
                float(gate.center[1] + world_offset[1]),
                float(gate.center[2] + world_offset[2]),
            )
            for gate in sequences[vehicle]
        ]
        client.simPlotPoints(
            centers,
            color_rgba=color,
            size=point_size,
            duration=duration,
            is_persistent=False,
        )
        print(
            f"  {vehicle} plotted with local-to-world offset "
            f"{_point(world_offset)}"
        )
        client.simPlotLineStrip(
            centers,
            color_rgba=color,
            thickness=max(1.0, point_size / 2.0),
            duration=duration,
            is_persistent=False,
        )
    print(f"Plotted non-persistent gate centers for {duration:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print/plot gate centers without changing the AirSim world."
    )
    parser.add_argument("--epsilon", type=float, choices=[0.0, 0.8, 1.6], default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-spacing", type=float, default=6.0)
    parser.add_argument("--gate-count", type=int, default=5)
    parser.add_argument("--x-start", type=float, default=10.0)
    parser.add_argument("--x-spacing", type=float, default=10.0)
    parser.add_argument("--base-z", type=float, default=-2.0)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--approach-distance", type=float, default=2.0)
    parser.add_argument("--exit-distance", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-duration", type=float, default=30.0)
    parser.add_argument("--point-size", type=float, default=20.0)
    args = parser.parse_args()

    sequences = generate_gate_sequences(
        VEHICLES,
        epsilon=args.epsilon,
        seed=args.seed,
        group_spacing=args.group_spacing,
        base_z=args.base_z,
        gate_count=args.gate_count,
        x_start=args.x_start,
        x_spacing=args.x_spacing,
        max_z=args.max_z,
    )
    _print_layout(
        sequences,
        approach_distance=args.approach_distance,
        exit_distance=args.exit_distance,
    )
    if args.plot:
        if args.plot_duration <= 0.0 or args.point_size <= 0.0:
            raise ValueError("plot duration and point size must be positive")
        _plot_layout(
            sequences,
            duration=args.plot_duration,
            point_size=args.point_size,
        )


if __name__ == "__main__":
    main()
