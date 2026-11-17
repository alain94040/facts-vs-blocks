"""Task 5b: a fair test for recurring heads (PLAN.md Phase 2b).

Uses the Task 5 head feature table (results/task5/heads.csv) without recomputing it, with the
Task 5 feature sets and standardisation (analysis/task5.py).

1. Gaussian null: for X against Y, fit mean and full covariance of Y's 48 standardised vectors
   (+ 1e-3 I); draw 48 null heads, 100 times (seed 0). A head of X recurs if its nearest real
   distance in Y is below the 5th percentile of its nearest distances to the null heads.
2. One-to-one matching with scipy.optimize.linear_sum_assignment on Euclidean distance: median
   matched distance, the same against each null draw (median and 5th percentile), fraction of
   matched pairs in the same layer (chance 1/6).
3. Head types: STORY features of the trained models pooled; k-means for k = 2..20 (10 restarts,
   seed 0) and mean silhouette (scipy.spatial.distance); the same on as many heads drawn from one
   Gaussian fitted to the pooled heads.

  python analysis/task5b.py --heads results/task5/heads.csv --task5 results/task5 --out results/task5b
  python analysis/task5b.py --pilot --heads results/task5_pilot/heads.csv --task5 results/task5_pilot \
      --out results/task5b_pilot
"""
import argparse
import csv
import json
import os
import shlex
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import task5 as T5  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_DRAWS = 100
RIDGE = 1e-3
KS = list(range(2, 21))
N_HEADS = 48

REAL = {"bio": T5.REAL["bio"], "nobio": T5.REAL["nobio"],
        "pairs": [(n, a, b, fs) for n, a, b, fs in T5.REAL["head_pairs"]] + [
            ("main-8k-s0 vs main-0-s0", "main-8k-s0", "main-0-s0", "STORY"),
            ("main-8k-s0 vs untrained (sanity)", "main-8k-s0", T5.UNTRAINED, "STORY")]}
PILOT = {"bio": T5.PILOT["bio"], "nobio": T5.PILOT["nobio"],
         "pairs": [("pilot P2 vs P1", "P2-main-2k-s0", "P1-main-2k-s0", "STORY"),
                   ("pilot P2 vs untrained (sanity)", "P2-main-2k-s0", T5.UNTRAINED, "STORY")]}


def read_heads(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if k in ("model",):
                    d[k] = v
                elif k in ("layer", "head"):
                    d[k] = int(v)
                elif k == "bio":
                    d[k] = v == "True"
                else:
                    d[k] = float(v) if v not in ("", "nan") else float("nan")
            rows.append(d)
    return rows


def gaussian(Z, rng, n, ridge=RIDGE):
    mu = Z.mean(0)
    cov = np.cov(Z, rowvar=False) + ridge * np.eye(Z.shape[1])
    w, V = np.linalg.eigh(cov)
    L = V * np.sqrt(np.clip(w, 0, None))
    return mu + rng.standard_normal((n, Z.shape[1])) @ L.T


def compare(Zx, Zy, rng):
    d_real = cdist(Zx, Zy)
    nearest = d_real.min(1)
    r, c = linear_sum_assignment(d_real)
    matched = d_real[r, c]
    null_near = np.zeros((N_DRAWS, len(Zx)))
    null_match_med = np.zeros(N_DRAWS)
    for s in range(N_DRAWS):
        Zn = gaussian(Zy, rng, len(Zy))
        dn = cdist(Zx, Zn)
        null_near[s] = dn.min(1)
        rn, cn = linear_sum_assignment(dn)
        null_match_med[s] = np.median(dn[rn, cn])
    thr = np.percentile(null_near, 5, axis=0)
    rec = nearest < thr
    return {"fraction_recur": float(rec.mean()), "recurs": rec.tolist(),
            "nearest_distance": nearest.tolist(), "null_5th_percentile": thr.tolist(),
            "median_nearest": float(np.median(nearest)),
            "median_null_5th_percentile": float(np.median(thr)),
            "one_to_one": {"median_matched_distance": float(np.median(matched)),
                           "null_median_of_medians": float(np.median(null_match_med)),
                           "null_5th_percentile_of_medians": float(np.percentile(null_match_med,
                                                                                 5)),
                           "fraction_same_layer": float(np.mean(r // 8 == c // 8)),
                           "chance_same_layer": 1 / 6,
                           "assignment": [int(x) for x in c]}}


def silhouette(D, lab):
    s = np.zeros(len(lab))
    for i in range(len(lab)):
        same = lab == lab[i]
        if same.sum() <= 1:
            s[i] = 0.0
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def sweep(Z, rng):
    D = squareform(pdist(Z))
    out = []
    for k in KS:
        _, lab, _ = T5.kmeans(Z, k, rng, n_init=10)
        out.append({"k": k, "silhouette": silhouette(D, lab), "labels": lab.tolist()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heads", default="results/task5/heads.csv")
    p.add_argument("--task5", default="results/task5")
    p.add_argument("--out", default="results/task5b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    cfg = PILOT if args.pilot else REAL
    out = os.path.join(T5.C.ROOT, args.out)
    os.makedirs(out, exist_ok=True)
    rows = read_heads(os.path.join(T5.C.ROOT, args.heads))
    trained = cfg["bio"] + cfg["nobio"]
    std = {"FULL": T5.standardiser([r for r in rows if r["model"] in cfg["bio"]],
                                   T5.FEATURES["FULL"]),
           "STORY": T5.standardiser([r for r in rows if r["model"] in trained],
                                    T5.FEATURES["STORY"])}
    with open(os.path.join(T5.C.ROOT, args.task5, "head_matching.json")) as f:
        old = json.load(f)["pairs"]
    rng = np.random.default_rng(args.seed)

    res = {"seed": args.seed, "ridge": RIDGE, "n_draws": N_DRAWS,
           "feature_sets": {k: v[1] for k, v in std.items()}, "pairs": {}}
    for name, x, y, fs in cfg["pairs"]:
        tf = std[fs][0]
        Zx = tf(sorted([r for r in rows if r["model"] == x], key=lambda r: (r["layer"], r["head"])))
        Zy = tf(sorted([r for r in rows if r["model"] == y], key=lambda r: (r["layer"], r["head"])))
        assert len(Zx) == len(Zy) == N_HEADS
        d = compare(Zx, Zy, rng)
        d.update({"x": x, "y": y, "feature_set": fs,
                  "old_null_fraction_recur": old[name]["fraction_recur"] if name in old else None})
        res["pairs"][name] = d
        print(name, fs, "recur", d["fraction_recur"], "old", d["old_null_fraction_recur"],
              "1:1", round(d["one_to_one"]["median_matched_distance"], 2),
              round(d["one_to_one"]["null_median_of_medians"], 2),
              round(d["one_to_one"]["null_5th_percentile_of_medians"], 2),
              "same layer", round(d["one_to_one"]["fraction_same_layer"], 3), flush=True)

    pool = sorted([r for r in rows if r["model"] in trained],
                  key=lambda r: (trained.index(r["model"]), r["layer"], r["head"]))
    Zp = std["STORY"][0](pool)
    tr = sweep(Zp, rng)
    Zn = gaussian(Zp, rng, len(Zp), ridge=0.0)
    nu = sweep(Zn, rng)
    best = max(tr, key=lambda r: r["silhouette"])
    clusters = []
    lab = np.array(best["labels"])
    for c in range(best["k"]):
        mem = [r for r, l in zip(pool, lab) if l == c]
        clusters.append({"cluster": c, "size": len(mem),
                         "mean_story_scores": {k: float(np.mean([r[k] for r in mem]))
                                               for k in T5.STORY_SCORES},
                         "heads_per_layer": np.bincount([r["layer"] for r in mem],
                                                        minlength=6).tolist(),
                         "n_models": len({r["model"] for r in mem}),
                         "models": sorted({r["model"] for r in mem})})
    res["head_types"] = {
        "n_heads": len(pool), "models": trained,
        "trained": [{"k": r["k"], "silhouette": r["silhouette"]} for r in tr],
        "null": [{"k": r["k"], "silhouette": r["silhouette"]} for r in nu],
        "best_k_trained": best["k"], "best_silhouette_trained": best["silhouette"],
        "null_silhouette_at_best_k": next(r["silhouette"] for r in nu if r["k"] == best["k"]),
        "best_silhouette_null": max(r["silhouette"] for r in nu),
        "max_abs_gap": max(abs(a["silhouette"] - b["silhouette"]) for a, b in zip(tr, nu)),
        "clusters_at_best_k": clusters}
    print("head types best k", best["k"], round(best["silhouette"], 3), "null at k",
          round(res["head_types"]["null_silhouette_at_best_k"], 3), flush=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")

    names = list(res["pairs"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, [res["pairs"][n]["old_null_fraction_recur"] or 0 for n in names], 0.4,
           label="Task 5 null (column shuffle)", color="0.6")
    ax.bar(x + 0.2, [res["pairs"][n]["fraction_recur"] for n in names], 0.4,
           label="Gaussian null", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({res['pairs'][n]['feature_set']})" for n in names], fontsize=7)
    ax.set_ylabel("fraction of heads that recur")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "recurrence.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(KS, [r["silhouette"] for r in tr], "o-", label="trained heads")
    ax.plot(KS, [r["silhouette"] for r in nu], "o--", color="0.5", label="Gaussian null")
    ax.set_xlabel("k")
    ax.set_ylabel("mean silhouette")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "silhouette.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
