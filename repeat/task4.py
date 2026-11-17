"""Repeat task 4: interchangeability on SEARCH, TEXT only (repeat/PLAN.md).

For layers i between 15% and 85% of depth (ceil(0.15 n) <= i <= floor(0.85 n)) and every j with
1 <= |i - j| <= 8 (0 <= j < n): swap(i, j), scored on the first 50 TEXT SEARCH chunks. Loss
increase = swap loss minus base loss on the same chunks, with a paired bootstrap interval.

  python repeat/task4.py --model main --seed 0
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_TEXT = 50
MAX_DIST = 8


def out_dir(model, task="task4"):
    d = os.path.join(ROOT, "results", "repeat", task)
    return d if model == "main" else os.path.join(d, model)


def pairs(n):
    lo, hi = math.ceil(0.15 * n), math.floor(0.85 * n)
    return [(i, j) for i in range(lo, hi + 1) for j in range(max(0, i - MAX_DIST),
                                                            min(n, i + MAX_DIST + 1)) if j != i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    out = out_dir(args.model)
    if args.replot:
        with open(os.path.join(out, "swap.csv")) as f:
            plot(list(csv.DictReader(f)), out)
        return
    torch.manual_seed(args.seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    t0 = time.time()
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    base = E.score(flow, sets, "SEARCH", which=("TEXT",), n_text=N_TEXT)["TEXT"]["loss"]
    pr = pairs(flow.n_layers)
    rows, losses = [], {}
    for k, (i, j) in enumerate(pr):
        sc = E.score(flow, sets, "SEARCH", {"kind": "swap", "start": i, "j": j}, which=("TEXT",),
                     n_text=N_TEXT)["TEXT"]["loss"]
        bd = E.bootstrap_diff(sc, base)
        rows.append({"i": i, "j": j, "dist": abs(i - j), "loss": float(sc.mean()),
                     "loss_increase": bd["diff"], "ci_low": bd["ci_low"],
                     "ci_high": bd["ci_high"]})
        losses[f"{i}_{j}"] = sc
        if k % 20 == 0:
            print(f"{k}/{len(pr)} swap({i},{j}) +{bd['diff']:.3f} {time.time() - t0:.0f}s",
                  flush=True)
    with open(os.path.join(out, "swap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(out, "swap_losses.npz"), base=base, **losses)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "n_layers": flow.n_layers,
           "n_text_chunks": N_TEXT, "base_loss": float(base.mean()), "n_pairs": len(rows),
           "i_range": [pr[0][0], pr[-1][0]], "max_dist": MAX_DIST,
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(rows, out)
    print(json.dumps(res, indent=1))


def swap_matrix(rows, n):
    H = np.full((n, n), np.nan)
    for r in rows:
        H[int(r["i"]), int(r["j"])] = float(r["loss_increase"])
    return H


def plot(rows, out):
    n = max(max(int(r["i"]), int(r["j"])) for r in rows) + 1
    H = swap_matrix(rows, n)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(np.clip(H, 1e-3, None)), cmap="viridis_r", origin="lower")
    ax.set_xlabel("j (layer whose weights are run)")
    ax.set_ylabel("i (position)")
    ax.set_title("swap(i, j): TEXT loss increase, log10 nats (values below 0.001 shown at 0.001; empty = not run)", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "swap_heat.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
