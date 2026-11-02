import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from airsim_process import (
    AirSimLaunchConfig,
    AirSimProcessError,
    AirSimProcessManager,
    discover_default_map,
)


PROJECT = Path(r"F:\Airsim\AirSim\Unreal\Environments\Blocks\Blocks.uproject")
EDITOR = Path(r"E:\epicGamesStrg\UE_4.27\Engine\Binaries\Win64\UE4Editor.exe")


class _FakeClient:
    def getServerVersion(self):
        return 1


class _FakeProcess:
    pid = 1234

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


class AirSimProcessTests(unittest.TestCase):
    def test_map_is_discovered_from_project_config(self):
        self.assertEqual(discover_default_map(PROJECT), "/Game/FalconEmpty.FalconEmpty")

    def test_command_uses_discovered_map_and_game_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary)
            manager = AirSimProcessManager(config)
            command = manager.build_command()
        self.assertIn("/Game/FalconEmpty.FalconEmpty", command)
        self.assertIn("-game", command)
        self.assertIn("-unattended", command)

    def test_windows_launch_detaches_long_running_game_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary)
            manager = AirSimProcessManager(config)
            with patch.object(manager, "_rpc_port_is_open", return_value=False), patch(
                "airsim_process.subprocess.Popen", return_value=_FakeProcess()
            ) as popen:
                manager.start()
                manager.stop()
            flags = popen.call_args.kwargs["creationflags"]
        expected = getattr(__import__("subprocess"), "DETACHED_PROCESS", 0)
        if expected:
            self.assertNotEqual(flags & expected, 0)

    def test_null_rhi_is_added_for_headless_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary, null_rhi=True)
            manager = AirSimProcessManager(config)
        self.assertIn("-nullrhi", manager.build_command())

    def test_runtime_rpc_timeout_is_longer_than_readiness_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary)
        self.assertGreaterEqual(config.rpc_timeout, 10.0)

    def test_readiness_polling_uses_a_real_client_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary)
            manager = AirSimProcessManager(config, client_factory=lambda: _FakeClient())
            fake_process = _FakeProcess()
            with patch("airsim_process.subprocess.Popen", return_value=fake_process):
                manager.start()
            try:
                client = manager.wait_until_ready(timeout=0.1, retry_interval=0.01)
                self.assertIsInstance(client, _FakeClient)
            finally:
                manager.stop()
            self.assertTrue(fake_process.terminated)

    def test_managed_launch_rejects_an_existing_rpc_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = AirSimLaunchConfig.from_paths(EDITOR, PROJECT, log_dir=temporary)
            manager = AirSimProcessManager(config)
            with patch.object(manager, "_rpc_port_is_open", return_value=True):
                with self.assertRaisesRegex(AirSimProcessError, "already in use"):
                    manager.start()


if __name__ == "__main__":
    unittest.main()
