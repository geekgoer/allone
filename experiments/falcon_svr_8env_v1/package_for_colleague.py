"""Create a minimal Windows acquisition ZIP, not an SD image or board runtime."""
from pathlib import Path
import argparse
from datetime import datetime
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(output, airsim_client):
    config = json.loads((HERE / 'collection_config_moderate_v2.json').read_text(encoding='utf-8'))
    benchmark = json.loads((ROOT / config['benchmark_config']).read_text(encoding='utf-8'))
    catalog = json.loads((ROOT / config['model_manifest']).read_text(encoding='utf-8'))
    files = {}

    def include(source, relative=None):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = relative or source.relative_to(ROOT)
        files[Path(relative).as_posix()] = source

    for name in ('airsim_interface.py', 'config.py', 'control_adapter.py', 'dnn_model_utils.py',
                 'environment.py', 'falcon12_fpga_udp_client.py', 'gate_route.py',
                 'gate_visualization.py', 'initial_state_generator.py', 'mpc_controller.py',
                 'reference_generator.py', 'state_adapter.py'):
        include(ROOT / name)
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.ps1', '.md', '.json', '.txt', '.png'):
            include(path)
    legacy = ROOT / 'experiments/paper_track_p_12d_5ms_v1/fpga_shadow_historical5ms'
    for name in ('run_fpga_closed_loop.py', 'run_fpga_shadow.py', 'startup_reference.py'):
        include(legacy / name)
    for name in (config['benchmark_config'], config['fpga_config'], config['model_manifest'],
                 benchmark['initial_states'], 'configs/thrust_calibration.json',
                 'configs/settings_three_drones.json',
                 'experiments/paper_track_p_12d_5ms_v1/physical_benchmark_v1/benchmark_utils.py'):
        include(ROOT / name)
    package = (ROOT / config['model_manifest']).parent
    for model in catalog['models']:
        for name in ('normalization.json', 'golden_vectors.npz', 'model_spec.json'):
            include(package / 'models' / model['name'] / 'export' / name)
    for path in sorted((airsim_client / 'airsim').glob('*.py')):
        include(path, Path('vendor/airsim') / path.name)
    if 'vendor/airsim/client.py' not in files:
        raise FileNotFoundError('AirSim source package not found under the selected PythonClient directory')
    include(airsim_client.parent / 'LICENSE', 'vendor/AirSim_LICENSE')
    manifest = dict(format_version=1, purpose='Eight-environment Windows SVR acquisition only',
        board_endpoint='192.168.77.2:5008 FLCN/v2',
        board_runtime_and_sd_image_included=False, airsim_simulator_included=False,
        training_weights_and_history_included=False, previous_results_included=False,
        sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, f'falcon_svr_colleague/{name}')
        archive.writestr('falcon_svr_colleague/transfer_manifest.json', json.dumps(manifest, indent=2) + '\n')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(json.dumps(dict(archive=str(output.resolve()), files=len(files), size_mib=round(output.stat().st_size/2**20, 2),
                          sha256=digest, note='Also transfer the full SD image and current5ms board-runtime archive separately.'), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'colleague_transfer' /
                        f"falcon_svr_colleague_{datetime.now():%Y%m%d_%H%M%S}.zip")
    parser.add_argument('--airsim-client', type=Path, default=Path('F:/Airsim/AirSim/PythonClient'))
    args = parser.parse_args()
    build(args.output.resolve(), args.airsim_client.resolve())


if __name__ == '__main__':
    main()
