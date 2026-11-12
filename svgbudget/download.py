"""Downloads, each pinned to an exact version so every run uses the same files."""
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

QUICKDRAW_REPO = "https://github.com/googlecreativelab/quickdraw-dataset"
QUICKDRAW_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"


def download_model(repo_id, root="models"):
    """Download a Hugging Face model pinned to its current commit, into models/<name>@<sha>."""
    sha = HfApi().model_info(repo_id).sha
    local_dir = Path(root) / f"{repo_id.split('/')[-1]}@{sha[:12]}"
    path = snapshot_download(repo_id, revision=sha, local_dir=local_dir)
    return {"repo": repo_id, "revision": sha, "local_path": str(path)}


def download_categories(root="prompts"):
    """The 345 Quick, Draw! category names, pinned to the repo's current git commit."""
    sha = subprocess.run(["git", "ls-remote", QUICKDRAW_REPO, "HEAD"],
                         capture_output=True, text=True, check=True).stdout.split()[0]
    url = f"https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/{sha}/categories.txt"
    out = Path(root) / f"quickdraw_categories@{sha[:12]}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, out)
    return {"source": QUICKDRAW_REPO, "revision": sha, "local_path": str(out)}


def download_sketches(category, k, recognized_only=True):
    """The first k sketches of one category. Streams the file and stops early (files are 100+ MB)."""
    url = QUICKDRAW_URL.format(urllib.parse.quote(category))
    sketches = []
    with urllib.request.urlopen(url) as resp:
        version = {h: resp.headers.get(h) for h in ("ETag", "Last-Modified")}
        for line in resp:
            d = json.loads(line)
            if recognized_only and not d["recognized"]:
                continue
            sketches.append({"key_id": d["key_id"], "countrycode": d["countrycode"], "drawing": d["drawing"]})
            if len(sketches) >= k:
                break
    return {"category": category, "url": url, **version, "sketches": sketches}
