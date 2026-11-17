"""Repeat task 2: blocks on SEARCH (repeat/PLAN.md).

Contiguous blocks of length 2, 4 and 8 starting at every second layer between 20% and 80% of
depth (even start layers s with 0.2 n <= s <= 0.8 n), keeping only blocks that fit in the model
(s + length <= n). Interventions: repeat and gain(2). All four sets, paired bootstrap against base.

Outputs results/repeat/task2/[<model>/]{blocks.csv, metrics.json, cmd.txt, scores/*.npz,
heat_repeat.png, heat_gain2.png}.

  python repeat/task2.py --model main --seed 0
"""
import argparse
import csv
import json
import math
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402
from repeat import flow as FL  # noqa: E402
from repeat.task1 import compare  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LENGTHS = [2, 4, 8]
INTERVENTIONS = [("repeat", {"kind": "repeat"}), ("gain2", {"kind": "gain", "g": 2.0})]


def out_dir(model):
    d = os.path.join(ROOT, "results", "repeat", "task2")
    return d if model == "main" else os.path.join(d, model)


def blocks(n):
    starts = [s for s in range(math.ceil(0.2 * n), math.floor(0.8 * n) + 1) if s % 2 == 0]
    return [(s, L) for L in LENGTHS for s in starts if s + L <= n], starts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    out = out_dir(args.model)
    if args.replot:
        with open(os.path.join(out, "blocks.csv")) as f:
            plots(list(csv.DictReader(f)), out)
        return
    torch.manual_seed(args.seed)
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    t0 = time.time()
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    base = E.score(flow, sets, "SEARCH")
    E.save_scores(os.path.join(out, "scores", "base.npz"), base)
    rows = [{"start": -1, "length": 0, "end": -1, "intervention": "base", **compare(base, base)}]
    bl, starts = blocks(flow.n_layers)
    for s, L in bl:
        for name, iv in INTERVENTIONS:
            sc = E.score(flow, sets, "SEARCH", {**iv, "start": s, "end": s + L})
            E.save_scores(os.path.join(out, "scores", f"S{s:02d}_L{L}_{name}.npz"), sc)
            rows.append({"start": s, "length": L, "end": s + L, "intervention": name,
                         **compare(sc, base)})
        print(f"block {s}+{L} done {time.time() - t0:.0f}s",
              {r["intervention"]: round(r["d_REASON_logprob"], 3) for r in rows[-2:]}, flush=True)
    with open(os.path.join(out, "blocks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "k": E.K_FROZEN,
           "n_layers": flow.n_layers, "starts": starts, "lengths": LENGTHS,
           "blocks": [list(b) for b in bl], "n_configurations": len(rows) - 1,
           "rule": "even start s with 0.2n <= s <= 0.8n and s + length <= n",
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    plots(rows, out)
    print(json.dumps(res, indent=1))


def plots(rows, out):
    rows = [r for r in rows if r["intervention"] != "base"]
    starts = sorted({int(r["start"]) for r in rows})
    vals = [float(r["d_REASON_logprob"]) for r in rows]
    vmax = max(abs(v) for v in vals) or 1.0
    for name, _ in INTERVENTIONS:
        H = np.full((len(LENGTHS), len(starts)), np.nan)
        sig = np.zeros_like(H, dtype=bool)
        for r in rows:
            if r["intervention"] == name:
                i, j = LENGTHS.index(int(r["length"])), starts.index(int(r["start"]))
                H[i, j] = float(r["d_REASON_logprob"])
                sig[i, j] = float(r["d_REASON_logprob_lo"]) > 0 or float(r["d_REASON_logprob_hi"]) < 0
        fig, ax = plt.subplots(figsize=(1 + 0.6 * len(starts), 3))
        im = ax.imshow(H, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if not np.isnan(H[i, j]):
                    ax.text(j, i, f"{H[i, j]:+.2f}" + ("*" if sig[i, j] else ""), ha="center",
                            va="center", fontsize=7)
        ax.set_xticks(range(len(starts)))
        ax.set_xticklabels(starts)
        ax.set_yticks(range(len(LENGTHS)))
        ax.set_yticklabels(LENGTHS)
        ax.set_xlabel("start layer")
        ax.set_ylabel("block length")
        ax.set_title(f"{name}: change in REASON log-prob (* = interval excludes 0)", fontsize=9)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"heat_{name}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
