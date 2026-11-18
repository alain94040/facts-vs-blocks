"""Middle task 6: back-patching (middle/PLAN.md, Part 2).

For every eligible TWO item (task 4; solved and failed reported separately) and every pair of
layers i < j, both even and in 2..n-2 (2..26 on 7B): the residual stream after layer j at the
last prompt token is written in place of the stream after layer i at that token, and layers
i+1..n-1 are run again (earlier positions are unchanged; causal attention). Success: the answer's
first token is the greedy top prediction.
Per (i, j): fraction of failed items fixed, fraction of solved items broken.
Control: the same with the source state from a different item of the same kind (same layer j;
one fixed random pairing, seed 0: a derangement of the eligible items).
Check: patching with j = i (the item's own state, unchanged) reproduces task 4's TWO result.

  python middle/task6.py --model dev --seed 0
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


@torch.no_grad()
def full_states(flow, seqs, layers):
    """Embedding-run once: full-sequence stream after each layer in `layers`, last-token stream
    after every layer, and the context; all for one batch."""
    x, last = H.pad(seqs)
    h = flow.inner.embed_tokens(x)
    ctx = flow._ctx(h, torch.arange(h.shape[1], device=h.device)[None])
    r = torch.arange(len(last), device=h.device)
    keep, lastst = {}, []
    for l in range(flow.n_layers):
        h = flow.layer(l, h, ctx)
        lastst.append(h[r, last])
        if l in layers:
            keep[l] = h.clone()
    return keep, torch.stack(lastst, 1), ctx, last, r


@torch.no_grad()
def patched_top(flow, h_i, i, src, ctx, last, r):
    h = h_i.clone()
    h[r, last] = src
    for l in range(i + 1, flow.n_layers):
        h = flow.layer(l, h, ctx)
    return flow.model.lm_head(flow.inner.norm(h[r, last])).float().argmax(-1)


def derangement(n, seed):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not (p == np.arange(n)).any():
            return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task6")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(C.out_dir(args.model, "task4"), "items.json")) as f:
        ok = json.load(f)
    m, tok = FL.load(C.MODELS[args.model])
    flow = FL.Flow(m)
    n = flow.n_layers
    L = list(range(2, n - 1, 2))
    its = H.items(tok)
    res = {}          # (kind, i, j, cond) -> bool array over items
    meta, check = {}, {}
    for name, (two, hop2) in H.TWO_KINDS.items():
        idx = [k for k, o in enumerate(ok) if o["HOP1"] and o[hop2]]
        T = [its[two][k] for k in idx]
        ans = torch.tensor([x["ans"] for x in T], device="mps")
        solved = np.array([ok[k][two] for k in idx])
        perm = derangement(len(idx), args.seed)
        meta[name] = {"n_eligible": len(idx), "n_solved": int(solved.sum()),
                      "n_failed": int((~solved).sum()), "solved": solved,
                      "control_same_answer_frac": float(np.mean(
                          [T[k]["answer"] == T[perm[k]]["answer"] for k in range(len(idx))]))}
        hs, lastst, ctx, last, r = full_states(flow, [x["ids"] for x in T], set(L))
        pt = torch.from_numpy(perm).to("mps")
        for i in L:
            top = patched_top(flow, hs[i], i, lastst[:, i], ctx, last, r)
            check[f"{name}_i{i}"] = bool(((top == ans).cpu().numpy() == solved).all())
            for j in L:
                if j <= i:
                    continue
                for cond, src in (("real", lastst[:, j]), ("control", lastst[pt, j])):
                    top = patched_top(flow, hs[i], i, src, ctx, last, r)
                    res[(name, i, j, cond)] = (top == ans).cpu().numpy()
            print(f"{time.time() - t0:.0f}s {name} i={i} done", flush=True)
        del hs
    rows = []
    for name in list(H.TWO_KINDS) + ["both"]:
        kinds = list(H.TWO_KINDS) if name == "both" else [name]
        sol = np.concatenate([meta[k]["solved"] for k in kinds])
        for i in L:
            for j in L:
                if j <= i:
                    continue
                row = {"kind": name, "i": i, "j": j, "n_failed": int((~sol).sum()),
                       "n_solved": int(sol.sum())}
                for cond in ("real", "control"):
                    s = np.concatenate([res[(k, i, j, cond)] for k in kinds])
                    pre = "" if cond == "real" else "control_"
                    row[pre + "fixed"] = float(s[~sol].mean()) if (~sol).any() else None
                    row[pre + "broken"] = float((~s[sol]).mean()) if sol.any() else None
                    row[pre + "n_fixed"] = int(s[~sol].sum())
                rows.append(row)
    with open(os.path.join(out, "backpatch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    z = C.zones(n)
    best = {}
    for name in list(H.TWO_KINDS) + ["both"]:
        R = [x for x in rows if x["kind"] == name and x["fixed"] is not None]
        if R:
            b = max(R, key=lambda x: (x["fixed"] - x["control_fixed"], x["fixed"]))
            bf = [x for x in R if x["i"] in z["FAMILY"]]
            bfam = max(bf, key=lambda x: (x["fixed"], -x["control_fixed"])) if bf else None
            best[name] = {"max_fixed_minus_control": b, "max_fixed_i_in_FAMILY": bfam,
                          "mean_fixed": float(np.mean([x["fixed"] for x in R])),
                          "mean_control_fixed": float(np.mean([x["control_fixed"] for x in R]))}
    summ = {"model": C.MODELS[args.model], "seed": args.seed, "layers": L, "zones": z,
            "items": {k: {a: b for a, b in v.items() if a != "solved"} for k, v in meta.items()},
            "identity_patch_reproduces_task4": all(check.values()), "identity_check": check,
            "best": best, "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(summ, f, indent=1)
    heat(rows, L, z, out)
    print(json.dumps({k: summ[k] for k in ("items", "identity_patch_reproduces_task4")}, indent=1))


def heat(rows, L, z, out):
    fig, axs = plt.subplots(2, 2, figsize=(11, 9))
    for row, (key, lab) in enumerate((("fixed", "failed items fixed"),
                                      ("broken", "solved items broken"))):
        for col, pre in enumerate(("", "control_")):
            A = np.full((len(L), len(L)), np.nan)
            for x in rows:
                if x["kind"] == "both" and x[pre + key] is not None:
                    A[L.index(x["i"]), L.index(x["j"])] = x[pre + key]
            ax = axs[row, col]
            im = ax.imshow(A, vmin=0, vmax=1, cmap="magma", origin="upper")
            ax.set_xticks(range(len(L)), L, fontsize=7)
            ax.set_yticks(range(len(L)), L, fontsize=7)
            ax.set_xlabel("source layer j")
            ax.set_ylabel("target layer i")
            ax.set_title(f"{'control: ' if pre else ''}{lab} (both kinds)", fontsize=10)
            for b in (z["FAMILY"][0], z["FAMILY"][-1]):
                if b in L:
                    ax.axhline(L.index(b), color="c", lw=0.5)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "backpatch.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
