# FALCON cold-RTI 12-D: three-largest-model server package

Models: 34x1024, 22x1024, 34x512. All start from seed 0, train exactly 2000 epochs, use fixed LR 1e-4, and have no early stopping or LR scheduler.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sha256sum -c SHA256SUMS.txt
python train_server_three_largest_2000.py --device cuda --batch-size 512 --workers 2
```

If the server runs out of GPU memory, use `--batch-size 256`. Re-run the same command after interruption; each model resumes from `latest_checkpoint.pt`.
