"""Middle task 5: where each step happens (middle/PLAN.md, Part 2).

For every eligible TWO item (task 4) and its HOP2 item, at the last prompt token, logit lens
(final norm + unembedding) on the residual stream after every layer l = 0..n-1. Rank = number of
vocabulary tokens with a strictly larger logit (0 = top). "After layer l in the top 10" = rank < 10.
  L_bridge   first l with the bridge (country) first token in the top 10, TWO prompt
  L_answer2  same for the answer, TWO prompt
  L_answer1  same for the answer, HOP2 prompt
  delay      L_answer2 - L_answer1 where both exist
Reuse: S(L) = sum of the updates of layers L-2, L-1, L = stream after L minus stream before L-2
(needs L >= 2). Cosine between S_HOP2(L_answer1) and S_TWO(L_answer2) of the same item, outlier
coordinates removed. Null: an item's TWO sum against the HOP2 sum of a different item with a
different answer (one fixed random pairing, seed 0: items are put in a seeded random cycle and
each item's partner is the next item after it with a different answer. A first version took the
first such item of the whole permutation, which gave nearly every item the same partner).

  python middle/task5.py --model dev --seed 0
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from middle import common as C  # noqa: E402
from middle import twohop as H  # noqa: E402
from repeat import flow as FL  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOPK = 10


def ranks(flow, st, tokens):
    """st: (n, n_layers + 1, d); returns (n, n_layers) ranks after each layer."""
    out = []
    for i in range(st.shape[0]):
        lg = H.lens_logits(flow, st[i, 1:])                      # (n_layers, V)
        t = lg[:, tokens[i]]
        out.append((lg > t[:, None]).sum(-1).cpu().numpy())
    return np.stack(out)


def first_top(r):
    w = np.where(r < TOPK)[0]
    return int(w[0]) if len(w) else None


def block_sum(st, L):
    return None if L is None or L < 2 else (st[L + 1] - st[L - 2]).double().numpy()


def cos(a, b, keep):
    a, b = a[keep], b[keep]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def boot_mean(x, seed=0, n=2000):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return {"mean": float(x.mean()), "ci_low": float(np.quantile(bs, 0.025)),
            "ci_high": float(np.quantile(bs, 0.975)), "n": int(len(x))}


def describe(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)), "mean": float(np.mean(v)),
            "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task5")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(C.out_dir(args.model, "task4"), "items.json")) as f:
        ok = json.load(f)
    m, tok = FL.load(C.MODELS[args.model])
    flow = FL.Flow(m)
    n_layers, d = flow.n_layers, flow.cfg.hidden_size
    z = C.zones(n_layers)
    keep = np.setdiff1d(np.arange(d), C.outliers(args.model))
    its = H.items(tok)
    rows, summary, rank_curves = [], {}, {}
    for name, (two, hop2) in H.TWO_KINDS.items():
        idx = [i for i, o in enumerate(ok) if o["HOP1"] and o[hop2]]
        T = [its[two][i] for i in idx]
        P = [its[hop2][i] for i in idx]
        sT = H.states(flow, [x["ids"] for x in T])
        sP = H.states(flow, [x["ids"] for x in P])
        rb = ranks(flow, sT, [x["bridge"] for x in T])
        ra2 = ranks(flow, sT, [x["ans"] for x in T])
        ra1 = ranks(flow, sP, [x["ans"] for x in P])
        rank_curves[name] = {"bridge_TWO": rb, "answer_TWO": ra2, "answer_HOP2": ra1,
                             "solved": np.array([ok[i][two] for i in idx])}
        Lb = [first_top(r) for r in rb]
        L2 = [first_top(r) for r in ra2]
        L1 = [first_top(r) for r in ra1]
        sums_T = [block_sum(sT[k], L2[k]) for k in range(len(idx))]
        sums_P = [block_sum(sP[k], L1[k]) for k in range(len(idx))]
        rng = np.random.default_rng(args.seed)
        perm = list(rng.permutation(len(idx)))
        partner = []
        for k in range(len(idx)):  # next suitable item after k in the seeded cycle
            pos = perm.index(k)
            c = [perm[(pos + s) % len(perm)] for s in range(1, len(perm))]
            c = [q for q in c if T[q]["answer"] != T[k]["answer"] and sums_P[q] is not None]
            partner.append(int(c[0]) if c else None)
        real, null = [], []
        for k in range(len(idx)):
            rc = (cos(sums_P[k], sums_T[k], keep)
                  if sums_P[k] is not None and sums_T[k] is not None else None)
            q = partner[k]
            nc = (cos(sums_P[q], sums_T[k], keep)
                  if q is not None and sums_T[k] is not None else None)
            o = ok[idx[k]]
            rows.append({"kind": name, "country": o["country"], "answer": T[k]["answer"],
                         "TWO_solved": o[two], "L_bridge": Lb[k], "L_answer2": L2[k],
                         "L_answer1": L1[k],
                         "delay": None if L1[k] is None or L2[k] is None else L2[k] - L1[k],
                         "reuse_cos": rc, "null_partner": None if q is None else T[q]["country"],
                         "null_cos": nc})
            if rc is not None and nc is not None:
                real.append(rc)
                null.append(nc)
        R = [r for r in rows if r["kind"] == name]
        summary[name] = {
            "n_eligible": len(idx), "n_solved": int(sum(r["TWO_solved"] for r in R)),
            "L_bridge": describe([r["L_bridge"] for r in R]),
            "L_answer2": describe([r["L_answer2"] for r in R]),
            "L_answer1": describe([r["L_answer1"] for r in R]),
            "delay": describe([r["delay"] for r in R]),
            "delay_solved": describe([r["delay"] for r in R if r["TWO_solved"]]),
            "delay_failed": describe([r["delay"] for r in R if not r["TWO_solved"]]),
            "frac_L_bridge_none": float(np.mean([r["L_bridge"] is None for r in R])),
            "frac_L_answer2_none": float(np.mean([r["L_answer2"] is None for r in R])),
            "frac_L_answer2_in_OUTPUT_of_found": float(np.mean(
                [r["L_answer2"] in z["OUTPUT"] for r in R if r["L_answer2"] is not None]))
            if any(r["L_answer2"] is not None for r in R) else None,
            "reuse_real": boot_mean(real) if real else None,
            "reuse_null": boot_mean(null, 1) if null else None,
            "reuse_real_minus_null": boot_mean(np.array(real) - np.array(null), 2)
            if real else None,
        }
        print(name, json.dumps({k: v for k, v in summary[name].items()
                                if k in ("n_eligible", "delay", "reuse_real", "reuse_null")}),
              flush=True)
    with open(os.path.join(out, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": C.MODELS[args.model], "seed": args.seed, "zones": z, "top_k": TOPK,
           "summary": summary, "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(rows, rank_curves, z, n_layers, out)


def plot(rows, curves, z, n, out):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for col, name in enumerate(H.TWO_KINDS):
        c = curves[name]
        ax = axs[0, col]
        for key, lab in (("bridge_TWO", "bridge, TWO"), ("answer_TWO", "answer, TWO"),
                         ("answer_HOP2", "answer, HOP2")):
            ax.plot(range(n), np.median(c[key], 0) + 1, "o-", ms=3, label=lab)
        ax.axhline(TOPK, color="k", lw=0.6, ls=":")
        ax.axvspan(z["FAMILY"][0] - 0.5, z["FAMILY"][-1] + 0.5, color="0.92", zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("layer (state after it)")
        ax.set_ylabel("median rank + 1")
        ax.set_title(f"{name}: logit lens, eligible items")
        ax.legend(fontsize=8)
        ax = axs[1, col]
        R = [r for r in rows if r["kind"] == name]
        bins = np.arange(-0.5, n + 0.5)
        for key in ("L_bridge", "L_answer1", "L_answer2"):
            ax.hist([r[key] for r in R if r[key] is not None], bins=bins, alpha=0.5, label=key)
        ax.set_xlabel("layer")
        ax.set_ylabel("items")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "lens.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in H.TWO_KINDS:
        R = [r for r in rows if r["kind"] == name and r["reuse_cos"] is not None
             and r["null_cos"] is not None]
        ax.hist([r["reuse_cos"] for r in R], bins=20, range=(-1, 1), alpha=0.5,
                label=f"{name} real")
        ax.hist([r["null_cos"] for r in R], bins=20, range=(-1, 1), alpha=0.5, histtype="step",
                lw=1.5, label=f"{name} null")
    ax.set_xlabel("cosine of three-layer update sums (outliers removed)")
    ax.set_ylabel("items")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "reuse.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
