"""Download model weights to data/cache/hf (git-ignored). Usage: python repeat/download.py <repo id>"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
from huggingface_hub import snapshot_download  # noqa: E402

for repo in sys.argv[1:]:
    p = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print(repo, p, flush=True)
