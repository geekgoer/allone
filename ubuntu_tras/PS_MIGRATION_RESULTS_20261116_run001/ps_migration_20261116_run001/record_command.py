#!/usr/bin/env python3
"""Preserve exact commands, combined output and exit codes; refuse overwrite."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--name',required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd: p.error('command required')
root=Path(__file__).resolve().parent/'logs'
root.mkdir(exist_ok=True)
log=root/(a.name+'.log')
meta=root/(a.name+'.json')
if log.exists() or meta.exists():raise RuntimeError('choose new command name')
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
log.write_bytes(result.stdout)
meta.write_text(json.dumps(dict(command=cmd,cwd=str(Path.cwd()),start_utc=start,
    end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=result.returncode),indent=2))
sys.stdout.buffer.write(result.stdout)
sys.stdout.flush()
sys.exit(result.returncode)
