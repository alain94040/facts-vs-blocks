"""Download model weights and transcoder weights (safetensors only) to data/cache/hf; record sizes.

Usage: python circuit/download.py <repo id> [<repo id> ...]
Appends one line per repo to results/circuit/task0/downloads.jsonl (repo, revision, bytes, free disk before/after).
"""
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.yaml"]
out = os.path.join(ROOT, "results", "circuit", "task0", "downloads.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
for repo in sys.argv[1:]:
    info = HfApi().model_info(repo, files_metadata=True)
    need = sum(s.size or 0 for s in info.siblings
               if s.rfilename.endswith((".json", ".safetensors", ".txt", ".model", ".yaml"))
               and not s.rfilename.startswith("features/"))
    free0 = shutil.disk_usage(ROOT).free
    print(repo, info.sha, f"need {need/1e9:.1f} GB, free {free0/1e9:.1f} GB", flush=True)
    if need > free0 - 20e9:
        sys.exit(f"not enough disk for {repo}")
    path = snapshot_download(repo, revision=info.sha, allow_patterns=PATTERNS)
    rec = {"repo": repo, "revision": info.sha, "bytes": need, "free_before": free0,
           "free_after": shutil.disk_usage(ROOT).free, "path": path}
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)
