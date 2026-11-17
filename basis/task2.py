"""Basis task 2: does the shared basis keep the model working (basis/PLAN.md).

Every FAMILY layer's matrices of one group (mlp: gate, up, down; attn: q, k, v, o) are projected
on their residual side, W -> U U^T W, with
  own(k)    : that layer's own top-k basis (eigenvectors of its folded W W^T, float64 CPU);
  shared(k) : the family's shared top-k basis (task 1, all coordinates);
  null(k)   : the rotated-null shared basis rotated back for each layer, Q_l^T U_null.
The projection is applied to the folded matrix and the norm gain divided out again, so the
model computes exactly with U U^T W_folded. Projections run on MPS in float32 and are written back
in float16; original weights are restored after every configuration. Evaluation: Experiment C's
SEARCH halves of TEXT, REASON and RECALL-HARD, paired bootstrap against base.

Parameter counts of the family for the group: original sum of d*n; own(k) per layer d*k + k*n;
shared(k) d*k once per type plus per layer k*n (null(k) counted as shared(k)).

  python basis/task2.py --model main --seed 0
  python basis/task2.py --model confirm --configs base own:4 shared:4 --seed 0     (task 3)
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
from basis import common as C  # noqa: E402
from basis.task1 import out_dir  # noqa: E402
from repeat import evaluate as E  # noqa: E402
from repeat import flow as FL  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SETS = ("TEXT", "REASON", "RECALL-HARD")
METRICS = [("TEXT", "loss"), ("REASON", "logprob"), ("REASON", "correct"),
           ("RECALL-HARD", "logprob"), ("RECALL-HARD", "correct")]
READ = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
        "gate": "mlp.gate_proj", "up": "mlp.up_proj"}
WRITE = {"o": "self_attn.o_proj", "down": "mlp.down_proj"}
DIVS = [8, 4, 2]


def module(layer, path):
    m = layer
    for part in path.split("."):
        m = getattr(m, part)
    return m


def gain(layer, t):
    return (layer.input_layernorm.weight if t in ("q", "k", "v")
            else layer.post_attention_layernorm.weight)


@torch.no_grad()
def project(layer, t, U):
    """Replace the matrix of type t by its residual-side projection on span(U) (U: d x k)."""
    U = torch.from_numpy(np.ascontiguousarray(U)).float().to(layer.input_layernorm.weight.device)
    if t in READ:
        w = module(layer, READ[t]).weight
        g = gain(layer, t).float()
        wf = w.float() * g[None, :]                  # folded, (n, d): residual side = columns
        new = ((wf @ U) @ U.T) / g[None, :]
    else:
        w = module(layer, WRITE[t]).weight            # (d, n): residual side = rows
        new = U @ (U.T @ w.float())
    w.copy_(new.to(w.dtype))


def own_bases(W, model, family, types, kmax):
    """Own top-kmax eigenvectors per family layer and type (cached as float32)."""
    cache = os.path.join(C.CACHE, model, "own")
    os.makedirs(cache, exist_ok=True)
    out = {}
    for l in family:
        M = None
        for t in types:
            p = os.path.join(cache, f"L{l:02d}_{t}.npy")
            if not os.path.exists(p):
                if M is None:
                    M = C.matrices(W, l)
                w = C.to64(M[t])
                np.save(p, C.top_eigvecs(w @ w.T)[0][:, :kmax].astype(np.float32))
            out[(l, t)] = p
    return out


def params(d, ns, family, k, scheme):
    if scheme == "base":
        return sum(d * n for n in ns.values()) * len(family)
    if scheme == "own":
        return sum(d * k + k * n for n in ns.values()) * len(family)
    return sum(d * k for _ in ns) + sum(k * n for n in ns.values()) * len(family)


def compare(sc, base):
    row = {}
    for s, k in METRICS:
        name = f"{s}_{'acc' if k == 'correct' else k}"
        bd = E.bootstrap_diff(sc[s][k].astype(np.float64), base[s][k].astype(np.float64))
        row[name] = float(sc[s][k].mean())
        row[f"d_{name}"], row[f"d_{name}_lo"], row[f"d_{name}_hi"] = \
            bd["diff"], bd["ci_low"], bd["ci_high"]
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dev", "main", "confirm"], default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--configs", nargs="*", default=None,
                   help="subset like base own:4 shared:4 (number = divisor of d)")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = out_dir(args.model, "task3" if args.model == "confirm" else "task2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd_task2.txt" if args.model == "confirm" else "cmd.txt"),
              "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    W = C.weights(args.model)
    d, family = W.D, C.zones(W.n_layers)["FAMILY"]
    cfgs = [("base", None)] + [(s, dv) for s in ("own", "shared", "null") for dv in DIVS]
    if args.configs:
        want = set(args.configs)
        cfgs = [c for c in cfgs if (c[0] if c[1] is None else f"{c[0]}:{c[1]}") in want]
    m, tok = FL.load(E.MODELS[args.model])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    layers = flow.layers
    ns = {"q": W.H * W.hd, "k": W.KV * W.hd, "v": W.KV * W.hd, "o": W.H * W.hd,
          "gate": W.cfg["intermediate_size"], "up": W.cfg["intermediate_size"],
          "down": W.cfg["intermediate_size"]}
    orig = {(l, t): (module(layers[l], READ.get(t) or WRITE[t]).weight.detach().cpu().clone())
            for l in family for t in C.TYPES}
    kmax = d // 2
    ownp = own_bases(W, args.model, family, C.TYPES,
                     kmax) if any(c[0] == "own" for c in cfgs) else {}
    cache = os.path.join(C.CACHE, args.model)
    shared = {t: np.load(os.path.join(cache, f"FAMILY_{t}_shared.npy")) for t in C.TYPES}
    null = {t: np.load(os.path.join(cache, f"FAMILY_{t}_null.npy")) for t in C.TYPES}
    rot = {}

    def restore():
        for (l, t), w in orig.items():
            module(layers[l], READ.get(t) or WRITE[t]).weight.data.copy_(w.to(w.dtype))

    base = None
    rows = []
    for scheme, dv in cfgs:
        groups = [None] if scheme == "base" else ["mlp", "attn"]
        for grp in groups:
            k = None if dv is None else d // dv
            if scheme != "base":
                for l in family:
                    if scheme == "null" and l not in rot:
                        rot[l] = C.rotation(l, d)
                    for t in C.GROUPS[grp]:
                        if scheme == "own":
                            U = np.load(ownp[(l, t)])[:, :k]
                        elif scheme == "shared":
                            U = shared[t][:, :k]
                        else:
                            U = rot[l].T @ null[t][:, :k]
                        project(layers[l], t, U)
            sc = E.score(flow, sets, "SEARCH", which=SETS)
            restore()
            if scheme == "base":
                base = sc
            gns = {t: ns[t] for t in (C.GROUPS[grp] if grp else C.TYPES)}
            orig_p = params(d, gns, family, None, "base")
            sch_p = params(d, gns, family, k, "own" if scheme == "own" else "shared") \
                if scheme != "base" else orig_p
            row = {"config": scheme, "group": grp or "all", "k": k or d,
                   "k_frac": (k / d) if k else 1.0, "family_params_original": orig_p,
                   "family_params_scheme": sch_p, "compression": orig_p / sch_p,
                   **compare(sc, base)}
            rows.append(row)
            print(f"{time.time() - t0:.0f}s {scheme} {grp} k={k} comp={row['compression']:.2f}"
                  f" dTEXT={row['d_TEXT_loss']:+.4f}", flush=True)
    name = "edits.csv"
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"model": E.MODELS[args.model], "seed": args.seed, "family": family, "d": d,
           "n_per_type": ns, "n_configurations": len(rows),
           "projection_precision": "bases float64 CPU; projection float32 MPS; stored float16",
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "edits_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    plot(rows, out)


def plot(rows, out):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, scheme in enumerate(["own", "shared", "null"]):
        for grp, ls in [("mlp", "-"), ("attn", "--")]:
            r = sorted([x for x in rows if x["config"] == scheme and x["group"] == grp],
                       key=lambda x: x["compression"])
            if r:
                ax.plot([x["compression"] for x in r], [x["d_TEXT_loss"] for x in r], ls,
                        marker="o", color=f"C{i}", label=f"{scheme} {grp}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("family compression ratio for the group (original / scheme parameters)")
    ax.set_ylabel("change in TEXT loss (nats)")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "text_vs_compression.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
