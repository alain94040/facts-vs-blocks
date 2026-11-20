"""Shared set-up for Experiment H (hop/PLAN.md): paths, environment, model loading, tokenization,
memory sampling. Every script imports this module first."""
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.join(ROOT, "circuit"))  # ct_compat
sys.path.insert(0, os.path.join(ROOT, "middle"))   # countries
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psutil  # noqa: E402
import torch  # noqa: E402

MODEL = "google/gemma-2-2b"
TRANSCODERS = "gemma"  # circuit-tracer 0.5.0 resolves this to mwhanna/gemma-scope-transcoders
RESULTS = os.path.join(ROOT, "results", "hop")
CACHE = os.path.join(ROOT, "data", "cache", "hop")
DEVICE = torch.device("mps")
DTYPE = torch.float32
SEED = 0


def out_dir(task, run=""):
    d = os.path.join(RESULTS, task, run) if run else os.path.join(RESULTS, task)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    return d


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def load_replacement_model(lazy_encoder=False):
    """ReplacementModel as hop/PLAN.md fixes it: float32, MPS, transcoders "gemma", library
    defaults otherwise (lazy decoder)."""
    import ct_compat  # noqa: F401  (mps:0 -> mps for safetensors)
    from circuit_tracer import ReplacementModel
    rm = ReplacementModel.from_pretrained(MODEL, TRANSCODERS, device=DEVICE, dtype=DTYPE,
                                         lazy_encoder=lazy_encoder)
    rm.eval()
    return rm


def tokenize(tok, text):
    """Plain text with <bos> (the tokenizer adds it), no chat template. Returns a list of ids."""
    ids = tok(text)["input_ids"]
    assert ids[0] == tok.bos_token_id, ids[:3]
    return ids


def free():
    import gc
    gc.collect()
    torch.mps.empty_cache()


class MemSampler(threading.Thread):
    """Peaks of process RSS, MPS driver-allocated memory and their sum, sampled every 0.5 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.peak = {"rss": 0, "mps_driver": 0, "sum": 0}
        self.stop = False

    def sample(self):
        rss = self.proc.memory_info().rss
        drv = torch.mps.driver_allocated_memory()
        for k, v in (("rss", rss), ("mps_driver", drv), ("sum", rss + drv)):
            self.peak[k] = max(self.peak[k], v)

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(0.5)

    def reset(self):
        self.peak = {k: 0 for k in self.peak}
        self.sample()

    def gb(self):
        return {k: round(v / 2**30, 2) for k, v in self.peak.items()}
