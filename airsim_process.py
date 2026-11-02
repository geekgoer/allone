"""Start and supervise an unattended UE4/AirSim game session.

The manager owns only the UE4 process that it starts.  It never uses GUI
automation and it does not terminate unrelated editor processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from typing import Any, Callable, Sequence

from airsim_interface import require_airsim


class AirSimProcessError(RuntimeError):
    """Raised when the managed UE4/AirSim session cannot become ready."""


def discover_default_map(uproject: str | Path) -> str:
    """Read the configured UE4 game map from ``DefaultEngine.ini``.

    The map is intentionally discovered from the project rather than guessed
    in Python.  For this project it resolves to
    ``/Game/FalconEmpty.FalconEmpty``.
    """

    project = Path(uproject).expanduser().resolve()
    ini_path = project.parent / "Config" / "DefaultEngine.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"UE4 engine config was not found: {ini_path}")
    pattern = re.compile(r"^\s*GameDefaultMap\s*=\s*(\S+)\s*$")
    for line in ini_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            value = match.group(1)
            if value.startswith("/Game/"):
                return value
            raise AirSimProcessError(
                f"GameDefaultMap is not a project asset path: {value} ({ini_path})"
            )
    raise AirSimProcessError(f"GameDefaultMap was not found in {ini_path}")


def _validate_project_asset_map(uproject: Path, map_asset: str) -> None:
    """Validate a /Game map against the corresponding Content/*.umap file."""

    if not map_asset.startswith("/Game/"):
        raise ValueError("map_asset must be an Unreal project asset path beginning with /Game/")
    asset_path = map_asset.removeprefix("/Game/").split(".", 1)[0]
    map_file = uproject.parent / "Content" / (asset_path.replace("/", os.sep) + ".umap")
    if not map_file.exists():
        raise FileNotFoundError(
            f"The configured map asset does not exist as a .umap file: {map_file}"
        )


@dataclass(frozen=True)
class AirSimLaunchConfig:
    """Paths and launch settings for one managed UE4 game process."""

    ue4_editor: Path
    uproject: Path
    map_asset: str
    log_dir: Path
    startup_timeout: float = 120.0
    retry_interval: float = 1.0
    rpc_host: str = "127.0.0.1"
    rpc_port: int = 41451
    rpc_timeout: float = 10.0
    res_x: int = 1280
    res_y: int = 720
    null_rhi: bool = False

    @classmethod
    def from_paths(
        cls,
        ue4_editor: str | Path,
        uproject: str | Path,
        *,
        map_asset: str | None = None,
        log_dir: str | Path = "logs",
        startup_timeout: float = 120.0,
        retry_interval: float = 1.0,
        rpc_host: str = "127.0.0.1",
        rpc_port: int = 41451,
        rpc_timeout: float = 10.0,
        res_x: int = 1280,
        res_y: int = 720,
        null_rhi: bool = False,
    ) -> "AirSimLaunchConfig":
        project = Path(uproject).expanduser().resolve()
        editor = Path(ue4_editor).expanduser().resolve()
        resolved_map = map_asset or discover_default_map(project)
        _validate_project_asset_map(project, resolved_map)
        if startup_timeout <= 0.0 or retry_interval <= 0.0 or rpc_timeout <= 0.0:
            raise ValueError("startup_timeout, retry_interval and rpc_timeout must be positive")
        if not rpc_host or rpc_port < 1 or rpc_port > 65535:
            raise ValueError("rpc_host and rpc_port are invalid")
        if res_x < 320 or res_y < 240:
            raise ValueError("res_x/res_y are too small for a reliable UE4 launch")
        return cls(
            ue4_editor=editor,
            uproject=project,
            map_asset=resolved_map,
            log_dir=Path(log_dir).expanduser().resolve(),
            startup_timeout=float(startup_timeout),
            retry_interval=float(retry_interval),
            rpc_host=str(rpc_host),
            rpc_port=int(rpc_port),
            rpc_timeout=float(rpc_timeout),
            res_x=int(res_x),
            res_y=int(res_y),
            null_rhi=bool(null_rhi),
        )


class AirSimProcessManager:
    """Own one unattended UE4/AirSim process and poll its RPC readiness."""

    def __init__(
        self,
        config: AirSimLaunchConfig,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory
        self._process: subprocess.Popen[Any] | None = None
        self._log_handle: Any = None
        self.log_path: Path | None = None

    @property
    def process(self) -> subprocess.Popen[Any] | None:
        return self._process

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    def build_command(self) -> list[str]:
        command = [
            str(self.config.ue4_editor),
            str(self.config.uproject),
            self.config.map_asset,
            "-game",
            "-windowed",
            f"-ResX={self.config.res_x}",
            f"-ResY={self.config.res_y}",
            "-nosplash",
            "-unattended",
            "-log",
            "-stdout",
            "-FullStdOutLogOutput",
            "-NoSound",
        ]
        if self.config.null_rhi:
            command.append("-nullrhi")
        return command

    def start(self) -> int:
        """Launch UE4 in game mode and return its process id."""

        if self._process is not None and self._process.poll() is None:
            raise AirSimProcessError(f"The managed UE4 process is already running (pid={self.pid})")
        if not self.config.ue4_editor.exists():
            raise FileNotFoundError(f"UE4Editor.exe was not found: {self.config.ue4_editor}")
        if not self.config.uproject.exists():
            raise FileNotFoundError(f"The UE4 project was not found: {self.config.uproject}")
        # A managed launch must never silently attach to a manually opened
        # AirSim instance.  Besides defeating process ownership, that instance
        # may contain spawned gates or other scene state which contaminates an
        # otherwise headless dataset collection.
        if self._client_factory is None and self._rpc_port_is_open():
            raise AirSimProcessError(
                f"AirSim RPC port {self.config.rpc_host}:{self.config.rpc_port} is already in use. "
                "Close the existing AirSim/UE4 session before using --auto-launch, or omit "
                "--auto-launch when intentionally connecting to that session."
            )
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_path = self.config.log_dir / f"ue4_{stamp}.log"
        self._log_handle = self.log_path.open("w", encoding="utf-8", errors="replace")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        # A full teacher collection can run for hours.  On Windows, keeping
        # UE4 attached to the shell/job that launched Python can terminate the
        # simulator when that job is recycled, even while the collector is
        # still alive.  Detach the child from that job but retain the Popen
        # handle so ``stop()`` still owns and terminates this exact process.
        if os.name == "nt":
            creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
        try:
            self._process = subprocess.Popen(
                self.build_command(),
                cwd=str(self.config.uproject.parent),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception:
            self._log_handle.close()
            self._log_handle = None
            raise
        return self._process.pid

    def _new_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        airsim = require_airsim()
        # AirSim's default RPC timeout is 3600 s.  That is unsuitable for a
        # readiness loop because a refused/dead server could block the whole
        # collector for an hour before our own timeout is reached.
        return airsim.MultirotorClient(
            ip=self.config.rpc_host,
            port=self.config.rpc_port,
            timeout_value=self.config.rpc_timeout,
        )

    def _rpc_port_is_open(self) -> bool:
        try:
            with socket.create_connection(
                (self.config.rpc_host, self.config.rpc_port),
                timeout=min(0.5, self.config.rpc_timeout),
            ):
                return True
        except OSError:
            return False

    def wait_until_ready(
        self,
        timeout: float | None = None,
        retry_interval: float | None = None,
    ) -> Any:
        """Poll AirSim RPC until a real server call succeeds."""

        timeout = self.config.startup_timeout if timeout is None else float(timeout)
        retry_interval = self.config.retry_interval if retry_interval is None else float(retry_interval)
        if timeout <= 0.0 or retry_interval <= 0.0:
            raise ValueError("timeout and retry_interval must be positive")
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            if self._process is not None:
                return_code = self._process.poll()
                if return_code is not None:
                    raise AirSimProcessError(
                        f"UE4 exited before AirSim RPC became ready (returncode={return_code}, "
                        f"log={self.log_path})"
                    )
            # Injected factories are used by offline tests and may represent
            # an in-process fake server, so the socket probe is only needed
            # for real AirSim clients.
            if self._client_factory is None and not self._rpc_port_is_open():
                time.sleep(min(retry_interval, max(0.0, deadline - time.monotonic())))
                continue
            try:
                client = self._new_client()
                # This is a real RPC request.  A TCP port being open alone is
                # not sufficient because the server may still be initializing.
                client.getServerVersion()
                return client
            except Exception as exc:  # RPC client errors vary by AirSim version.
                last_error = exc
                time.sleep(min(retry_interval, max(0.0, deadline - time.monotonic())))
        raise AirSimProcessError(
            f"AirSim RPC did not become ready after {attempts} attempts in {timeout:.1f}s; "
            f"last_error={last_error!r}; ue4_log={self.log_path}"
        )

    def start_and_wait(self, timeout: float | None = None) -> Any:
        self.start()
        try:
            return self.wait_until_ready(timeout=timeout)
        except Exception:
            self.stop()
            raise

    def restart(self, timeout: float | None = None) -> Any:
        self.stop()
        return self.start_and_wait(timeout=timeout)

    def stop(self, wait_timeout: float = 10.0) -> None:
        """Stop only the UE4 process created by this manager."""

        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=wait_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=wait_timeout)
        self._process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> "AirSimProcessManager":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.stop()
