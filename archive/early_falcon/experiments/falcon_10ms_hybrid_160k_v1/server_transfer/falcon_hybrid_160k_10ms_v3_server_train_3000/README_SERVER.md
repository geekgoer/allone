# FALCON 10 m/s hybrid 160k — server training package

This package contains a prepared 12-D train/validation/test split and the six-model fixed-3000 runner.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server-training.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python train_six_models_server_3000.py --device cuda
```

Default batch size is 2048 for an approximately 11 GB NVIDIA GPU. If CUDA reports out of memory, restart with `--batch-size 1024`; all completed epochs are retained in `latest_checkpoint.pt`.
