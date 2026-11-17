"""Repeat task 1: single-layer map on SEARCH (repeat/PLAN.md).

For every layer: repeat, gain(2), gain(0), attn2, mlp2, scored on all four sets (TEXT, REASON,
RECALL, RECALL-HARD). Each configuration is compared with base on the same items (paired
bootstrap, 1,000 resamples, seed 0). Per-item scores are kept in scores/ for later tasks.

Outputs results/repeat/task1/<model tag>/{layers.csv, metrics.json, cmd.txt, scores/*.npz,
change_<set>.png}. For the main model the tag directory is omitted (results/repeat/task1/).

  python repeat/task1.py --model main --seed 0
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INTERVENTIONS = [("repeat", {"kind": "repeat"}), ("gain2", {"kind": "gain", "g": 2.0}),
                 ("gain0", {"kind": "gain", "g": 0.0}), ("attn2", {"kind": "attn2"}),
                 ("mlp2", {"kind": "mlp2"})]
# per set: (per-item key, summary name); for TEXT lower is better
METRICS = [("TEXT", "loss"), ("REASON", "logprob"), ("REASON", "correct"),
           ("RECALL", "logprob"), ("RECALL", "correct"), ("RECALL-HARD", "logprob"),
           ("RECALL-HARD", "correct")]


def out_dir(model):
    d = os.path.join(ROOT, "results", "repeat", "task1")
    return d if model == "main" else os.path.join(d, model)


def compare(sc, base):
    row = {}
    for s, k in METRICS:
        a = sc[s][k].astype(np.float64)
        b = base[s][k].astype(np.float64)
        name = f"{s}_{'acc' if k == 'correct' else k}"
        row[name] = float(a.mean())
        bd = E.bootstrap_diff(a, b)
        row[f"d_{name}"] = bd["diff"]
        row[f"d_{name}_lo"] = bd["ci_low"]
        row[f"d_{name}_hi"] = bd["ci_high"]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = out_dir(args.model)
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    t0 = time.time()
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    base = E.score(flow, sets, "SEARCH")
    E.save_scores(os.path.join(out, "scores", "base.npz"), base)
    base_row = {"layer": -1, "intervention": "base", **compare(base, base)}
    rows = [base_row]
    print("base", {k: v for k, v in base_row.items() if not k.startswith("d_")},
          flush=True)
    layers = args.layers if args.layers is not None else list(range(flow.n_layers))
    for l in layers:
        for name, iv in INTERVENTIONS:
            sc = E.score(flow, sets, "SEARCH", {**iv, "start": l})
            E.save_scores(os.path.join(out, "scores", f"L{l:02d}_{name}.npz"), sc)
            row = {"layer": l, "intervention": name, **compare(sc, base)}
            rows.append(row)
        print(f"layer {l} done {time.time() - t0:.0f}s", {r["intervention"]: round(
            r["d_REASON_logprob"], 3) for r in rows[-5:]}, flush=True)
    with open(os.path.join(out, "layers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "k": E.K_FROZEN,
           "n_layers": flow.n_layers, "n_configurations": len(rows) - 1,
           "base_search": {k: v for k, v in base_row.items() if not k.startswith("d_")
                           and k not in ("layer", "intervention")},
           "n_search_items": {s: len(sets["splits"][s]["SEARCH"]) for s in E.SETS},
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    plots(rows, out)
    print(json.dumps(res, indent=1))


def plots(rows, out):
    for s, k in [("TEXT", "loss"), ("REASON", "logprob"), ("RECALL", "logprob"),
                 ("RECALL-HARD", "logprob")]:
        fig, axs = plt.subplots(1, 2 if s != "TEXT" else 1, figsize=(13 if s != "TEXT" else 7, 4),
                                squeeze=False)
        panels = [(k, f"change in {s} {k}")]
        if s != "TEXT":
            panels.append(("acc", f"change in {s} accuracy"))
        for ax, (key, title) in zip(axs[0], panels):
            for i, (name, _) in enumerate(INTERVENTIONS):
                r = [x for x in rows if x["intervention"] == name]
                L = [x["layer"] for x in r]
                y = np.array([x[f"d_{s}_{key}"] for x in r])
                lo = np.array([x[f"d_{s}_{key}_lo"] for x in r])
                hi = np.array([x[f"d_{s}_{key}_hi"] for x in r])
                ax.plot(L, y, "o-", ms=3, color=f"C{i}", label=name)
                ax.fill_between(L, lo, hi, color=f"C{i}", alpha=0.15)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("layer")
            ax.set_title(title + (" (lower is better)" if s == "TEXT" else ""))
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"change_{s}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
