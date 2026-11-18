"""Middle task 2: one linear operator per layer (middle/PLAN.md, Part 1).

Convention: z (d+1) and u (d) are row vectors, the map is u ~ z M with M of shape (d+1, d)
(rows = input coordinates incl. the constant, columns = target coordinates).

Ridge on FIT (= FIT_A + FIT_B): M = (Szz + lam I)^-1 Szu, lam = c tr(Szz)/(d+1),
c in {1e-4, 1e-3, 1e-2, 1e-1} chosen per layer by TEST R^2. Everything is exact from the sums:
  err(M) = tr(Suu) - 2 tr(M^T Szu) + tr(M^T Szz M),  R^2 = 1 - err/tot,
  tot = tr(Suu) - n |mean u|^2   (mean u = constant row of Szu / n).
Two settings: all target coordinates, and the 16 outlier coordinates removed from the target
(columns of u). lam depends only on Szz, so the no-outlier map for a given c is the same map
with those columns dropped; c is chosen separately for each setting, by that setting's TEST R^2.

Noise ceiling: with the chosen c, fit on FIT_A and score on FIT_B's sums, and the reverse.

The chosen maps are saved to data/cache/middle/<model>/M<l>_<setting>.npy for task 3 (and, for
the no-outlier setting, the same c's map with all target columns, M<l>_no_outliers_full.npy).

  python middle/task2.py --model dev --seed 0
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from middle import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CS = [1e-4, 1e-3, 1e-2, 1e-1]
SETTINGS = ["all", "no_outliers"]


def keep_cols(d, out, setting):
    return np.arange(d) if setting == "all" else np.setdiff1d(np.arange(d), out)


def restrict(a, cols):
    """Sums with the target restricted to the given columns."""
    return {"Szz": a["Szz"], "Szu": a["Szu"][:, cols], "Suu": a["Suu"][np.ix_(cols, cols)],
            "n": a["n"]}


def tot(a):
    n = float(a["n"])
    mu = a["Szu"][-1] / n
    return float(np.trace(a["Suu"]) - n * mu @ mu)


def err_parts(M, a):
    """Column sums of the terms of err: (tr(Suu) scalar, tr(M^T Szu), tr(M^T Szz M))."""
    return (float(np.trace(a["Suu"])), float(np.einsum("ij,ij->", M, a["Szu"])),
            float(np.einsum("ij,ij->", M, a["Szz"] @ M)))


def r2(M, a):
    s, c1, c2 = err_parts(M, a)
    return 1 - (s - 2 * c1 + c2) / tot(a)


def ridge(a, c):
    D = a["Szz"].shape[0]
    lam = c * np.trace(a["Szz"]) / D
    return np.linalg.solve(a["Szz"] + lam * np.eye(D), a["Szu"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    np.random.seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(C.out_dir(args.model, "task1"), "metrics.json")) as f:
        n_layers, d = (lambda m: (m["n_layers"], m["d"]))(json.load(f))
    outl = C.outliers(args.model)
    z = C.zones(n_layers)
    rows = []
    for l in range(n_layers):
        A, B, T = (C.load_acc(args.model, l, s) for s in ("FIT_A", "FIT_B", "TEST"))
        F = {k: A[k] + B[k] for k in A}
        Ms = {c: ridge(F, c) for c in CS}
        for s in SETTINGS:
            cols = keep_cols(d, outl, s)
            Ts, Fs = restrict(T, cols), restrict(F, cols)
            test = {c: r2(Ms[c][:, cols], Ts) for c in CS}
            c = max(CS, key=lambda c: test[c])
            M = Ms[c][:, cols]
            np.save(os.path.join(C.CACHE, args.model, f"M{l:02d}_{s}.npy"), M)
            if s != "all":  # full target side at this c, for task 3's rotated null
                np.save(os.path.join(C.CACHE, args.model, f"M{l:02d}_{s}_full.npy"), Ms[c])
            MA, MB = ridge(A, c)[:, cols], ridge(B, c)[:, cols]
            ab = r2(MA, restrict(B, cols))
            ba = r2(MB, restrict(A, cols))
            rows.append({"layer": l, "zone": C.zone_of(l, z), "setting": s, "c": c,
                         "R2_ll": test[c], "R2_fit": r2(M, Fs),
                         **{f"R2_test_c{c_:g}": test[c_] for c_ in CS},
                         "ceiling_A_on_B": ab, "ceiling_B_on_A": ba,
                         "ceiling_mean": (ab + ba) / 2, "tot_test": tot(Ts),
                         "n_fit": float(F["n"]), "n_test": float(T["n"])})
            print(f"{time.time() - t0:.0f}s layer {l} {s}: c={c:g} R2_ll={test[c]:.3f} "
                  f"ceiling={(ab + ba) / 2:.3f}", flush=True)
    with open(os.path.join(out, "layers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summ = {}
    for s in SETTINGS:
        for zn, ls in z.items():
            r = [x for x in rows if x["setting"] == s and x["layer"] in ls]
            summ[f"{zn}_{s}"] = {"R2_ll_mean": float(np.mean([x["R2_ll"] for x in r])),
                                 "R2_ll_min": float(np.min([x["R2_ll"] for x in r])),
                                 "ceiling_mean": float(np.mean([x["ceiling_mean"] for x in r]))}
    res = {"model": C.MODELS[args.model], "seed": args.seed, "d": d, "n_layers": n_layers,
           "zones": z, "outliers": outl.tolist(), "cs": CS, "by_zone": summ,
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, s in zip(ax, SETTINGS):
        r = [x for x in rows if x["setting"] == s]
        a.plot([x["layer"] for x in r], [x["R2_ll"] for x in r], "o-", label="own map, TEST R²")
        a.plot([x["layer"] for x in r], [x["ceiling_mean"] for x in r], "s--",
               label="noise ceiling (half on half)")
        for zn, ls in z.items():
            if zn == "FAMILY":
                a.axvspan(ls[0] - 0.5, ls[-1] + 0.5, color="0.9", zorder=0)
        a.set_title(s)
        a.set_xlabel("layer")
        a.set_ylim(min(0, min(x["R2_ll"] for x in rows)), 1)
    ax[0].set_ylabel("R²")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r2_per_layer.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
