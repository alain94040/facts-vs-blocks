"""Experiment E shared code (middle/PLAN.md).

Models, zones, outlier coordinates, output and cache paths, and the wikitext-103 *train* chunks
used by Part 1.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402,F401  (sets HF_HOME through repeat.flow)
from repeat import flow as FL  # noqa: E402

MODELS = {"dev": FL.DEV_MODEL, "main": FL.MAIN_MODEL}
CACHE = os.path.join(ROOT, "data", "cache", "middle")

N_CHUNKS, CHUNK_LEN, FIRST_POS = 480, 256, 8
SPLITS = {"FIT_A": range(0, 190), "FIT_B": range(190, 380), "TEST": range(380, 480)}


def zones(n):
    """Experiment C zones: ENTRY 0-5, FAMILY 6-17, OUTPUT 18-27 on 7B (28 layers); the same
    fractions of depth, rounded, on other depths (0.5B: 0-4, 5-14, 15-23)."""
    if n == 28:
        a, b = 6, 18
    else:
        a, b = round(0.21 * n), round(0.64 * n)
    return {"ENTRY": list(range(0, a)), "FAMILY": list(range(a, b)), "OUTPUT": list(range(b, n))}


def zone_of(l, z):
    return next(k for k, v in z.items() if l in v)


def outliers(model):
    p = {"dev": os.path.join(ROOT, "results", "repeat", "task4b", "dev", "summary.json"),
         "main": os.path.join(ROOT, "results", "repeat", "task4b", "summary.json")}[model]
    with open(p) as f:
        return np.array(json.load(f)["outliers"])


def out_dir(model, task):
    d = os.path.join(ROOT, "results", "middle", task)
    return d if model == "main" else os.path.join(d, model)


def train_ids(tok, log=print):
    """Token ids of the whole wikitext-103-raw-v1 train split, cached (uint32). The text is
    tokenized in blocks of 2,000 lines joined as one string (lines keep their own newlines), so
    only the ~900 block boundaries differ from tokenizing the joined corpus in one call. All
    Qwen2.5 sizes share one tokenizer."""
    p = os.path.join(CACHE, "wikitext103_train_ids.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = ds["text"]
    blocks = ["".join(text[i:i + 2000]) for i in range(0, len(text), 2000)]
    ids = []
    for b0 in range(0, len(blocks), 64):
        for x in tok(blocks[b0:b0 + 64]).input_ids:
            ids.extend(x)
        log(f"tokenized {min(b0 + 64, len(blocks))}/{len(blocks)} blocks")
    arr = np.array(ids, dtype=np.uint32)
    os.makedirs(CACHE, exist_ok=True)
    np.save(p, arr)
    return arr


def chunks(tok, seed=0, log=print):
    """480 chunks of 256 tokens at random non-overlapping offsets of the train split (seed),
    in corpus order (as repeat/data.text_chunks does); chunk index = position in that order."""
    ids = train_ids(tok, log)
    n = len(ids) // CHUNK_LEN
    pick = np.sort(np.random.default_rng(seed).choice(n, N_CHUNKS, replace=False))
    out = np.stack([np.asarray(ids[i * CHUNK_LEN:(i + 1) * CHUNK_LEN]) for i in pick])
    return out.astype(np.int64), {"n_tokens": int(len(ids)), "n_chunks_available": int(n),
                                  "chunk_offsets": pick.tolist()}


def acc_path(model, l, split):
    return os.path.join(CACHE, model, f"L{l:02d}_{split}.npz")


def load_acc(model, l, split):
    """Sums for one layer and split; split 'FIT' = FIT_A + FIT_B."""
    if split == "FIT":
        a, b = load_acc(model, l, "FIT_A"), load_acc(model, l, "FIT_B")
        return {k: a[k] + b[k] for k in a}
    with np.load(acc_path(model, l, split)) as f:
        return {k: f[k] for k in f.files}
