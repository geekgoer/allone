"""Create and remove visible gate frames in an AirSim world.

The gate coordinates in :mod:`environment` are deliberately independent of
AirSim.  This module is the live-world adapter: it uses four scaled cube
meshes per gate, so the same generated gate centers are visible in UE4 and
remain collision-aware during the flight.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

import numpy as np

from airsim_interface import require_airsim
from environment import Gate


DEFAULT_GATE_ASSET = "1M_Cube"
DEFAULT_GATE_WIDTH = 4.0
DEFAULT_GATE_HEIGHT = 4.0
DEFAULT_GATE_THICKNESS = 0.15
DEFAULT_OBJECT_PREFIX = "FalconGate_"

# AirSim applies these values as Unreal component scale, not as requested
# dimensions in metres. Blocks' 1M_Cube is a one-metre mesh. UE4.27's
# EditorCube spans two metres in this project, so it needs half the component
# scale to produce the same physical bar dimensions. This also prevents the
# visually overlong "#"-shaped frame seen with uncorrected EditorCube bars.
ASSET_SCALE_MULTIPLIERS = {
    "1M_Cube": 1.0,
    "EditorCube": 0.5,
}


def _normalise_quaternion(value: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(value), dtype=np.float64).reshape(-1)
    if quaternion.size != 4:
        raise ValueError("quaternion must have shape (4,) in [w, x, y, z] order")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("quaternion cannot be zero")
    return quaternion / norm


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotate_vector(quaternion: np.ndarray, vector: Iterable[float]) -> np.ndarray:
    q = _normalise_quaternion(quaternion)
    v = np.asarray(list(vector), dtype=np.float64).reshape(-1)
    if v.size != 3:
        raise ValueError("vector must have shape (3,)")
    pure = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    conjugate = np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)
    return _quaternion_multiply(_quaternion_multiply(q, pure), conjugate)[1:]


def _airsim_pose(airsim: Any, position: np.ndarray, orientation: np.ndarray) -> Any:
    # AirSim's Python Quaternionr constructor is (x, y, z, w), while the
    # project uses the conventional numeric order [w, x, y, z].
    quaternion = _normalise_quaternion(orientation)
    return airsim.Pose(
        position_val=airsim.Vector3r(float(position[0]), float(position[1]), float(position[2])),
        orientation_val=airsim.Quaternionr(
            x_val=float(quaternion[1]),
            y_val=float(quaternion[2]),
            z_val=float(quaternion[3]),
            w_val=float(quaternion[0]),
        ),
    )


@dataclass(frozen=True)
class GateObject:
    """One visible cube making up a gate frame."""

    name: str
    gate_index: int
    part: str
    position: np.ndarray
    scale: np.ndarray


class SpawnedGateScene:
    """Handles the lifetime of objects spawned for one gate sequence."""

    def __init__(self, client: Any, objects: Iterable[GateObject]) -> None:
        self.client = client
        self.objects = tuple(objects)
        self._destroyed = False

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.objects)

    def destroy(self) -> None:
        if self._destroyed:
            return
        for item in reversed(self.objects):
            try:
                self.client.simDestroyObject(item.name)
            except Exception:
                # Cleanup is best effort; the flight exception should remain
                # the useful error if the simulator is already shutting down.
                pass
        self._destroyed = True

    def __enter__(self) -> "SpawnedGateScene":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.destroy()


def cleanup_generated_gates(client: Any, prefix: str = DEFAULT_OBJECT_PREFIX) -> list[str]:
    """Remove stale objects from an earlier run with our own name prefix."""

    list_objects = getattr(client, "simListSceneObjects", None)
    if not callable(list_objects):
        return []
    removed: list[str] = []
    for name in list_objects():
        if str(name).startswith(prefix):
            try:
                if client.simDestroyObject(name):
                    removed.append(str(name))
            except Exception:
                pass
    return removed


def _gate_parts(
    gate: Gate,
    width: float,
    height: float,
    thickness: float,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    center = np.asarray(gate.center, dtype=np.float64)
    orientation = _normalise_quaternion(gate.orientation)
    half_width = width / 2.0
    half_height = height / 2.0
    local_parts = (
        ("left", [0.0, -half_width, 0.0], [thickness, thickness, height]),
        ("right", [0.0, half_width, 0.0], [thickness, thickness, height]),
        ("top", [0.0, 0.0, -half_height], [thickness, width, thickness]),
        ("bottom", [0.0, 0.0, half_height], [thickness, width, thickness]),
    )
    return [
        (
            part,
            center + _rotate_vector(orientation, offset),
            np.asarray(scale, dtype=np.float64),
        )
        for part, offset, scale in local_parts
    ]


def spawn_gate_sequence(
    client: Any,
    gates: Iterable[Gate],
    *,
    width: float = DEFAULT_GATE_WIDTH,
    height: float = DEFAULT_GATE_HEIGHT,
    thickness: float = DEFAULT_GATE_THICKNESS,
    asset_name: str = DEFAULT_GATE_ASSET,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    clear_previous: bool = True,
    world_offset: Iterable[float] | None = None,
) -> SpawnedGateScene:
    """Spawn four static cube bars for every gate and return a cleanup handle.

    AirSim's object registry uses the short asset name (for example
    ``1M_Cube``), not the UE4 content-browser path.  The mesh is one metre
    wide by default, so the scale values below are expressed in metres.
    """

    if width <= thickness or height <= thickness or thickness <= 0.0:
        raise ValueError("width and height must exceed positive thickness")
    airsim = require_airsim()
    resolved_asset_name = str(asset_name)
    list_assets = getattr(client, "simListAssets", None)
    if callable(list_assets):
        available_assets = {str(asset) for asset in list_assets()}
        if resolved_asset_name not in available_assets:
            if resolved_asset_name == "1M_Cube" and "EditorCube" in available_assets:
                resolved_asset_name = "EditorCube"
                print(
                    "AirSim asset '1M_Cube' is unavailable in this game-mode registry; "
                    "using size-corrected 'EditorCube' for visible gates.",
                    flush=True,
                )
            else:
                sample = ", ".join(sorted(available_assets)[:12])
                raise RuntimeError(
                    f"AirSim asset '{resolved_asset_name}' is not registered in the running world. "
                    f"Available assets ({len(available_assets)}): {sample}"
                )
    if clear_previous:
        cleanup_generated_gates(client, object_prefix)
    offset = np.zeros(3, dtype=np.float64)
    if world_offset is not None:
        offset = np.asarray(world_offset, dtype=np.float64).reshape(-1)
        if offset.size != 3 or not np.all(np.isfinite(offset)):
            raise ValueError("world_offset must contain three finite values")

    spawned: list[GateObject] = []
    scale_multiplier = ASSET_SCALE_MULTIPLIERS.get(resolved_asset_name, 1.0)
    try:
        for gate in gates:
            world_gate = Gate(gate.index, gate.center + offset, gate.orientation)
            for part, position, scale in _gate_parts(world_gate, width, height, thickness):
                component_scale = scale * scale_multiplier
                requested_name = f"{object_prefix}{gate.index}_{part}"
                pose = _airsim_pose(airsim, position, gate.orientation)
                returned_name = client.simSpawnObject(
                    requested_name,
                    resolved_asset_name,
                    pose,
                    airsim.Vector3r(
                        float(component_scale[0]),
                        float(component_scale[1]),
                        float(component_scale[2]),
                    ),
                    False,
                    False,
                )
                actual_name = str(returned_name or requested_name)
                if not actual_name:
                    raise RuntimeError(
                        f"AirSim did not return an object name for {requested_name}"
                    )
                spawned.append(
                    GateObject(
                        name=actual_name,
                        gate_index=gate.index,
                        part=part,
                        position=position.copy(),
                        scale=component_scale.copy(),
                    )
                )
    except Exception as exc:
        SpawnedGateScene(client, spawned).destroy()
        raise RuntimeError(
            f"Could not spawn visible gates using AirSim asset '{resolved_asset_name}'. "
            "Run client.simListAssets() and choose an available static mesh. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return SpawnedGateScene(client, spawned)


def spawn_gate_sequences(
    client: Any,
    gate_sequences: Mapping[str, Iterable[Gate]],
    *,
    width: float = DEFAULT_GATE_WIDTH,
    height: float = DEFAULT_GATE_HEIGHT,
    thickness: float = DEFAULT_GATE_THICKNESS,
    asset_name: str = DEFAULT_GATE_ASSET,
    object_prefix: str = DEFAULT_OBJECT_PREFIX,
    clear_previous: bool = True,
    vehicle_world_offsets: Mapping[str, Iterable[float]] | None = None,
) -> SpawnedGateScene:
    """Spawn independent visible gate groups for multiple vehicles.

    If ``gate_sequences`` contains three vehicles with five gates each, this
    function creates 15 independent gate frames (60 cube bars).  Names are
    prefixed with the vehicle id, for example
    ``FalconGate_Drone1_1_left``.
    """

    if not gate_sequences:
        raise ValueError("gate_sequences must contain at least one vehicle")
    if clear_previous:
        cleanup_generated_gates(client, object_prefix)
    offsets = vehicle_world_offsets or {}
    all_objects: list[GateObject] = []
    try:
        for vehicle_name, gates in gate_sequences.items():
            safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", str(vehicle_name)).strip("_")
            if not safe_name:
                raise ValueError("vehicle names must contain letters, digits, or underscores")
            offset = np.asarray(offsets.get(vehicle_name, np.zeros(3)), dtype=np.float64).reshape(-1)
            if offset.size != 3 or not np.all(np.isfinite(offset)):
                raise ValueError(
                    f"world offset for {vehicle_name!r} must contain three finite values"
                )
            world_gates = [
                Gate(
                    gate.index,
                    np.asarray(gate.center, dtype=np.float64) + offset,
                    gate.orientation,
                )
                for gate in gates
            ]
            scene = spawn_gate_sequence(
                client,
                world_gates,
                width=width,
                height=height,
                thickness=thickness,
                asset_name=asset_name,
                object_prefix=f"{object_prefix}{safe_name}_",
                clear_previous=False,
            )
            all_objects.extend(scene.objects)
    except Exception:
        SpawnedGateScene(client, all_objects).destroy()
        raise
    return SpawnedGateScene(client, all_objects)
