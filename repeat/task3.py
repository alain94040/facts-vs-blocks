"""Repeat task 3: dose on SEARCH (repeat/PLAN.md).

The single layer (task 1) and the block (task 2) with the largest REASON log-probability gain
under repeat. For each: partial(a), a = 0.25, 0.5, 0.75, 1, and gain(g), g = 1.25, 1.5, 2.
All four sets, paired bootstrap against base.

  python repeat/task3.py --model main --seed 0
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402
from repeat import flow as FL  # noqa: E402
from repeat import task1 as T1  # noqa: E402
from repeat import task2 as T2  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PARTIAL = [0.25, 0.5, 0.75, 1.0]
GAIN = [1.25, 1.5, 2.0]


def out_dir(model):
    d = os.path.join(ROOT, "results", "repeat", "task3")
    return d if model == "main" else os.path.join(d, model)


def best(path, keys):
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["intervention"] == "repeat"]
    b = max(rows, key=lambda r: float(r["d_REASON_logprob"]))
    return {k: int(b[k]) for k in keys}, float(b["d_REASON_logprob"]), \
        float(b["d_REASON_logprob"]) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = out_dir(args.model)
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    t0 = time.time()
    lay, lay_gain, lay_pos = best(os.path.join(T1.out_dir(args.model), "layers.csv"), ["layer"])
    blk, blk_gain, blk_pos = best(os.path.join(T2.out_dir(args.model), "blocks.csv"),
                                  ["start", "end"])
    targets = {"layer": {"start": lay["layer"], "end": lay["layer"] + 1},
               "block": {"start": blk["start"], "end": blk["end"]}}
    sel = {"layer": {**targets["layer"], "repeat_d_REASON_logprob": lay_gain,
                     "positive": lay_pos},
           "block": {**targets["block"], "repeat_d_REASON_logprob": blk_gain,
                     "positive": blk_pos}}
    print("selected", sel, flush=True)
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    base = E.score(flow, sets, "SEARCH")
    rows = []
    for tname, t in targets.items():
        cfgs = [("partial", a, {"kind": "partial", "a": a}) for a in PARTIAL] + \
               [("gain", g, {"kind": "gain", "g": g}) for g in GAIN]
        for kind, dose, iv in cfgs:
            sc = E.score(flow, sets, "SEARCH", {**iv, **t})
            E.save_scores(os.path.join(out, "scores", f"{tname}_{kind}{dose}.npz"), sc)
            rows.append({"target": tname, "start": t["start"], "end": t["end"], "kind": kind,
                         "dose": dose, **T1.compare(sc, base)})
            print(tname, kind, dose, round(rows[-1]["d_REASON_logprob"], 3), flush=True)
    with open(os.path.join(out, "dose.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "selected": sel,
           "n_configurations": len(rows), "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(rows, out)
    print(json.dumps(res, indent=1))


def plot(rows, out):
    panels = [("REASON", "logprob"), ("REASON", "acc"), ("TEXT", "loss"),
              ("RECALL-HARD", "logprob")]
    fig, axs = plt.subplots(1, len(panels), figsize=(18, 4))
    for ax, (s, k) in zip(axs, panels):
        for i, tname in enumerate(["layer", "block"]):
            for kind, ls in [("partial", "-"), ("gain", "--")]:
                r = [x for x in rows if x["target"] == tname and x["kind"] == kind]
                # x axis: total step size relative to one pass; partial(a) -> 1 + a, gain(g) -> g
                x = [(1 + x_["dose"]) if kind == "partial" else x_["dose"] for x_ in r]
                y = [x_[f"d_{s}_{k}"] for x_ in r]
                lo = [x_[f"d_{s}_{k}_lo"] for x_ in r]
                hi = [x_[f"d_{s}_{k}_hi"] for x_ in r]
                lab = f"{tname} {r[0]['start']}-{r[0]['end'] - 1} {kind}"
                ax.plot(x, y, ls, marker="o", color=f"C{i}", label=lab)
                ax.fill_between(x, lo, hi, color=f"C{i}", alpha=0.1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("1 + a (partial) or g (gain)")
        ax.set_title(f"change in {s} {k}" + (" (lower is better)" if s == "TEXT" else ""))
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "dose.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
