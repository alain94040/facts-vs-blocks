"""Middle task 3: does one layer's operation explain another's (middle/PLAN.md, Part 1).

Maps M_l from task 2 (u ~ z M, M of shape (d+1, d) in the all-coordinates setting, (d+1, d-16)
without the outlier targets). For every pair (i, j), on layer i's TEST sums:
  R2_ij         TEST R^2 of M_j;
  R2_ij_scaled  of a M_j with the best scalar a = tr(M^T Szu) / tr(M^T Szz M) (closed form);
  ratio         R2_ij_scaled / R2_ii.
Null, rotated operator: Q (d x d random orthogonal, seed 0) applied to the residual block,
M_res -> Q M_res Q^T, the constant row left as is (taken before removing outlier targets; the
no-outlier null drops the same target columns afterwards). One Q for every j.

Read / write subspaces: the plan writes the operator as u = M z with "read" = right and
"write" = left singular vectors. In this file's convention (u = z M) that is: read = top-64
directions on the input side (left singular vectors of M_res), write = top-64 on the target
side (right singular vectors of M_res). Computed as top eigenvectors of M_res M_res^T and
M_res^T M_res (float64). Overlap = mean cosine of the principal angles = mean singular value of
U_i^T U_j. Null: layer j's subspaces taken from its rotated operator.

  python middle/task3.py --model dev --seed 0
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
from middle.task2 import SETTINGS, keep_cols, restrict, tot  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K_SUB = 64


def rotation(d, seed=0):
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))[None, :]


def rotate(M_full, Q):
    out = M_full.copy()
    out[:-1] = Q @ M_full[:-1] @ Q.T
    return out


def top(G, k):
    w, V = np.linalg.eigh(G)
    return V[:, ::-1][:, :k].copy()


def overlap(U, V):
    return float(np.linalg.svd(U.T @ V, compute_uv=False).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true", help="redraw heat maps from pairs.csv")
    args = p.parse_args()
    t0 = time.time()
    out = C.out_dir(args.model, "task3")
    os.makedirs(out, exist_ok=True)
    if args.replot:
        with open(os.path.join(out, "pairs.csv")) as f:
            table = [{k: (v if k in ("setting", "zone_i", "zone_j") else
                          (int(v) if k in ("i", "j") else float(v))) for k, v in r.items()}
                     for r in csv.DictReader(f)]
        n = max(r["i"] for r in table) + 1
        for s in SETTINGS:
            heatmaps(table, s, n, C.zones(n), out)
        return
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(C.out_dir(args.model, "task2"), "metrics.json")) as f:
        m2 = json.load(f)
    n_layers, d = m2["n_layers"], m2["d"]
    outl = C.outliers(args.model)
    z = C.zones(n_layers)
    Q = rotation(d, args.seed)
    cache = os.path.join(C.CACHE, args.model)

    def log(m):
        print(f"{time.time() - t0:.0f}s {m}", flush=True)

    rows = {}
    for s in SETTINGS:
        cols = keep_cols(d, outl, s)
        # maps: real, and the rotated null (rotated with the full target side, then restricted)
        real, null = {}, {}
        for j in range(n_layers):
            Mj = np.load(os.path.join(cache, f"M{j:02d}_{s}.npy"))
            real[j] = Mj
            full = full_map(cache, j, s, d, cols)
            null[j] = rotate(full, Q)[:, cols]
        log(f"{s}: maps loaded")
        # subspaces
        sub = {}
        for j in range(n_layers):
            for kind, M in (("real", real[j]), ("null", null[j])):
                R = M[:-1]
                sub[(j, kind)] = (top(R @ R.T, K_SUB), top(R.T @ R, K_SUB))
        log(f"{s}: subspaces done")
        for i in range(n_layers):
            T = restrict(C.load_acc(args.model, i, "TEST"), cols)
            tt, suu = tot(T), float(np.trace(T["Suu"]))
            for kind, maps in (("real", real), ("null", null)):
                for j in range(n_layers):
                    M = maps[j]
                    c1 = float(np.einsum("ij,ij->", M, T["Szu"]))
                    c2 = float(np.einsum("ij,ij->", M, T["Szz"] @ M))
                    r2 = 1 - (suu - 2 * c1 + c2) / tt
                    a = c1 / c2
                    r2s = 1 - (suu - 2 * a * c1 + a * a * c2) / tt
                    rd = overlap(sub[(i, "real")][0], sub[(j, kind)][0])
                    wr = overlap(sub[(i, "real")][1], sub[(j, kind)][1])
                    rows[(s, i, j, kind)] = {"r2": r2, "r2_scaled": r2s, "a": a,
                                             "read": rd, "write": wr}
            log(f"{s}: layer {i} done")
    table = []
    for s in SETTINGS:
        for i in range(n_layers):
            r2ii = rows[(s, i, i, "real")]["r2"]
            for j in range(n_layers):
                r, nl = rows[(s, i, j, "real")], rows[(s, i, j, "null")]
                table.append({
                    "setting": s, "i": i, "j": j, "zone_i": C.zone_of(i, z),
                    "zone_j": C.zone_of(j, z), "R2_ii": r2ii,
                    "R2_ij": r["r2"], "a": r["a"], "R2_ij_scaled": r["r2_scaled"],
                    "ratio": r["r2_scaled"] / r2ii, "read_overlap": r["read"],
                    "write_overlap": r["write"],
                    "null_R2_ij": nl["r2"], "null_a": nl["a"], "null_R2_ij_scaled": nl["r2_scaled"],
                    "null_ratio": nl["r2_scaled"] / r2ii, "null_read_overlap": nl["read"],
                    "null_write_overlap": nl["write"]})
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    res = {"model": C.MODELS[args.model], "seed": args.seed, "zones": z, "k_subspace": K_SUB,
           "summary": summarise(table, z), "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    for s in SETTINGS:
        heatmaps(table, s, n_layers, z, out)
    log("done")


def full_map(cache, j, s, d, cols):
    """The setting's map (its own c) with the full target side, saved by task 2."""
    if s == "all":
        return np.load(os.path.join(cache, f"M{j:02d}_all.npy"))
    return np.load(os.path.join(cache, f"M{j:02d}_{s}_full.npy"))


def mean_where(table, s, key, cond):
    v = [r[key] for r in table if r["setting"] == s and cond(r)]
    return float(np.mean(v)) if v else None


def summarise(table, z):
    fam, outp = set(z["FAMILY"]), set(z["OUTPUT"])
    out = {}
    for s in SETTINGS:
        S = {}
        for name, zone in (("FAMILY", fam), ("OUTPUT", outp), ("ENTRY", set(z["ENTRY"]))):
            inz = lambda r, zone=zone: r["i"] in zone and r["j"] in zone and r["i"] != r["j"]
            S[name] = {k: mean_where(table, s, k, inz) for k in
                       ("ratio", "null_ratio", "read_overlap", "null_read_overlap",
                        "write_overlap", "null_write_overlap", "R2_ij", "a")}
            S[name]["by_distance"] = {}
            for dist in range(1, len(zone)):
                c = lambda r, zone=zone, dist=dist: (r["i"] in zone and r["j"] in zone
                                                     and abs(r["i"] - r["j"]) == dist)
                S[name]["by_distance"][dist] = {
                    k: mean_where(table, s, k, c) for k in
                    ("ratio", "null_ratio", "read_overlap", "write_overlap")}
            near = lambda r, zone=zone: (r["i"] in zone and r["j"] in zone
                                         and 1 <= abs(r["i"] - r["j"]) <= 4)
            S[name]["ratio_upto4"] = mean_where(table, s, "ratio", near)
            S[name]["null_ratio_upto4"] = mean_where(table, s, "null_ratio", near)
            S[name]["ratio_min_pair"] = min(r["ratio"] for r in table if r["setting"] == s
                                            and inz(r))
        out[s] = S
    return out


def heatmaps(table, s, n, z, out):
    fig, axs = plt.subplots(3, 2, figsize=(10, 14))
    for row, (key, lab, lo) in enumerate([("ratio", "R²_ij,scaled / R²_ii", 0),
                                          ("read_overlap", "read overlap", 0),
                                          ("write_overlap", "write overlap", 0)]):
        for col, pre in enumerate(["", "null_"]):
            A = np.full((n, n), np.nan)
            for r in table:
                if r["setting"] == s:
                    A[r["i"], r["j"]] = r[pre + key]
            ax = axs[row, col]
            im = ax.imshow(np.clip(A, lo, 1), vmin=lo, vmax=1, cmap="viridis", origin="upper")
            for b in (z["FAMILY"][0] - 0.5, z["FAMILY"][-1] + 0.5):
                ax.axhline(b, color="w", lw=0.6)
                ax.axvline(b, color="w", lw=0.6)
            ax.set_title(f"{'null: ' if pre else ''}{lab} [{s}], colours clipped to [0, 1]",
                         fontsize=9)
            ax.set_xlabel("j (operator)")
            ax.set_ylabel("i (data)")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"heatmaps_{s}.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
