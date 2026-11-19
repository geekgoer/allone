"""Mark an episode range for clean recollection after simulator contamination.

The command is a dry run unless ``--apply`` is supplied. Existing episode
files are retained as evidence; their manifest entries become pending, so the
resumable collector atomically replaces them before a final merge can occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_store import atomic_write_json  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "paper_core_40k_acado_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--reason", default="clean recollection after wrong AirSim instance")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_count = int(manifest["episode_count"])
    if args.start < 0 or args.end < args.start or args.end >= episode_count:
        raise ValueError(
            f"episode range must satisfy 0 <= start <= end < {episode_count}; "
            f"received {args.start}..{args.end}"
        )

    ids = list(range(args.start, args.end + 1))
    completed = sum(
        manifest["episodes"][f"episode_{episode_id:04d}"]["status"] == "completed"
        for episode_id in ids
    )
    print(
        f"Prepared range: {args.start}..{args.end} ({len(ids)} episodes; "
        f"{completed} currently completed)."
    )
    if not args.apply:
        print("Dry run only. Re-run with --apply to mark this range pending.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = root / f"manifest.before_recollect_{args.start}_{args.end}_{stamp}.json"
    shutil.copy2(manifest_path, backup)
    for episode_id in ids:
        entry = manifest["episodes"][f"episode_{episode_id:04d}"]
        entry.update(
            status="pending",
            attempts=0,
            sample_count=0,
            last_error=f"scheduled for recollection: {args.reason}",
        )
    now = datetime.now(timezone.utc).isoformat()
    manifest.update(status="running", updated_at=now)
    manifest.pop("failed_episode_ids", None)
    manifest.pop("final_dataset", None)
    manifest.pop("statistics", None)
    manifest.setdefault("recollection_history", []).append(
        {
            "timestamp": now,
            "start_episode": args.start,
            "end_episode": args.end,
            "reason": args.reason,
            "manifest_backup": backup.name,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(f"Marked {len(ids)} episodes pending. Manifest backup: {backup}")
    print("Existing episode files were retained and will be atomically replaced on resume.")


if __name__ == "__main__":
    main()
