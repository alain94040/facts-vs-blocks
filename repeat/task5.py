"""Repeat task 5: confirm on REPORT (repeat/PLAN.md). First and only use of the REPORT halves.

Configurations (selected on SEARCH in tasks 1-3, before REPORT was seen):
  layer L = task 3's single layer, block B = task 3's block;
  repeat on L and B; gain(2) on L and B; attn2 and mlp2 on L;
  the best dose of task 3 for L and for B (largest REASON log-probability gain on SEARCH);
  B under partial(0.25) (added to the plan after task 3);
  reference: repeat and gain(2) on the layer at 50% of depth.
Every configuration on all four sets, paired bootstrap against base (1,000 resamples, seed 0).
Contrasts: repeat - gain(2) for L and B (QB); attn2 - mlp2 on L (QC).

  python repeat/task5.py --model main --seed 0
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
from repeat import task3 as T3  # noqa: E402
from repeat.task1 import METRICS  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def out_dir(model):
    d = os.path.join(ROOT, "results", "repeat", "task5")
    return d if model == "main" else os.path.join(d, model)


def configs(model, n_layers):
    with open(os.path.join(T3.out_dir(model), "metrics.json")) as f:
        sel = json.load(f)["selected"]
    with open(os.path.join(T3.out_dir(model), "dose.csv")) as f:
        dose = list(csv.DictReader(f))
    L, B = sel["layer"], sel["block"]
    lt = {"start": L["start"], "end": L["end"]}
    bt = {"start": B["start"], "end": B["end"]}

    def best_dose(target):
        r = max([x for x in dose if x["target"] == target], key=lambda x: float(x["d_REASON_logprob"]))
        kind, d = r["kind"], float(r["dose"])
        return (f"{kind}({d:g})", {"kind": kind, ("a" if kind == "partial" else "g"): d})

    mid = round(0.5 * n_layers)
    ld, ldiv = best_dose("layer")
    bd, bdiv = best_dose("block")
    cfg = [
        ("layer", "repeat", {"kind": "repeat", **lt}),
        ("layer", "gain(2)", {"kind": "gain", "g": 2.0, **lt}),
        ("layer", "attn2", {"kind": "attn2", **lt}),
        ("layer", "mlp2", {"kind": "mlp2", **lt}),
        ("layer", ld + " best dose", {**ldiv, **lt}),
        ("block", "repeat", {"kind": "repeat", **bt}),
        ("block", "gain(2)", {"kind": "gain", "g": 2.0, **bt}),
        ("block", bd + " best dose", {**bdiv, **bt}),
        ("block", "partial(0.25)", {"kind": "partial", "a": 0.25, **bt}),
        ("mid", "repeat", {"kind": "repeat", "start": mid, "end": mid + 1}),
        ("mid", "gain(2)", {"kind": "gain", "g": 2.0, "start": mid, "end": mid + 1}),
    ]
    return cfg, {"layer": lt, "block": bt, "mid_layer": mid, "layer_best_dose": ld,
                 "block_best_dose": bd}


def diff_row(a, b):
    row = {}
    for s, k in METRICS:
        name = f"{s}_{'acc' if k == 'correct' else k}"
        bd = E.bootstrap_diff(a[s][k].astype(np.float64), b[s][k].astype(np.float64))
        row[name] = float(a[s][k].mean())
        row[f"d_{name}"], row[f"d_{name}_lo"], row[f"d_{name}_hi"] = \
            bd["diff"], bd["ci_low"], bd["ci_high"]
        row[f"d_{name}_excl0"] = bd["excludes_zero"]
    return row


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
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    cfg, sel = configs(args.model, flow.n_layers)
    base = E.score(flow, sets, "REPORT")
    E.save_scores(os.path.join(out, "scores", "base.npz"), base)
    sc, rows = {}, []
    for target, name, iv in cfg:
        key = f"{target}:{name}"
        sc[key] = E.score(flow, sets, "REPORT", iv)
        E.save_scores(os.path.join(out, "scores", key.replace(":", "_").replace(" ", "_")
                                   .replace("(", "").replace(")", "") + ".npz"), sc[key])
        rows.append({"target": target, "config": name, "start": iv["start"], "end": iv["end"],
                     "vs": "base", **diff_row(sc[key], base)})
        print(key, round(rows[-1]["d_REASON_logprob"], 3), flush=True)
    for target, a, b in [("layer", "repeat", "gain(2)"), ("block", "repeat", "gain(2)"),
                         ("layer", "attn2", "mlp2"), ("mid", "repeat", "gain(2)")]:
        rows.append({"target": target, "config": f"{a} - {b}", "start": "", "end": "",
                     "vs": b, **diff_row(sc[f"{target}:{a}"], sc[f"{target}:{b}"])})
    with open(os.path.join(out, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "split": "REPORT",
           "selection": sel, "base_report": {f"{s}_{'acc' if k == 'correct' else k}":
                                             float(base[s][k].mean()) for s, k in METRICS},
           "n_report_items": {s: len(sets["splits"][s]["REPORT"]) for s in E.SETS},
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
