"""Run folders: every script writes into runs/<YYYYMMDD-HHMMSS>_<name>/ with its settings and environment."""
import json
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def start_run(name, config, root="runs"):
    """Create runs/<stamp>_<name>/ and record config.json, meta.json (git commit, command, GPUs) and env.txt."""
    run_dir = Path(root) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {
        "name": name,
        "started": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git_commit": _sh(["git", "rev-parse", "HEAD"]),
        # only tracked code counts as "dirty"; new run folders and notes don't
        "git_dirty": bool(_sh(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!docs"])),
        "host": socket.gethostname(),
        "gpus": _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).splitlines(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "env.txt").write_text(_sh([sys.executable, "-m", "pip", "freeze"]))
    if meta["git_dirty"]:
        print("[runlog] WARNING: uncommitted changes; this run can't be tied to an exact commit")
    print(f"[runlog] run dir: {run_dir}")
    return run_dir


def latest_run(name, root="runs"):
    """The most recent run folder named exactly <stamp>_<name> (so 'sft_v1' never matches 'eval_sft_v1')."""
    pattern = re.compile(rf"^\d{{8}}-\d{{6}}_{re.escape(name)}$")
    found = sorted(p for p in Path(root).iterdir() if p.is_dir() and pattern.match(p.name))
    if not found:
        raise FileNotFoundError(f"no run folder named *_{name} in {root}/")
    return found[-1]
