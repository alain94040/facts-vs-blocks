"""Repeat task 4c: a fair null for the weight comparison (repeat/PLAN.md). Weights only.

Reuses task 4b's folding, outlier set, similarity and matching code; task 4b's outputs are not
touched.

Vector types per layer (after folding the RMSNorm gains): MLP gate rows, up rows, down columns;
attention q rows, k rows, v rows and o columns (all in residual coordinates).

1. Covariance-matched null: for each layer and vector type, the mean (float64, CPU) and the
   covariance of that layer's vectors; as many null vectors as the layer has, drawn from that
   Gaussian (seed 0, per layer). Draws use x = mean + z^T Xc / sqrt(n - 1) with z standard normal
   and Xc the centred vectors, which is exactly the Gaussian with the sample covariance and needs
   no factorisation of the (for k and v, singular) covariance. A null neuron takes one draw of
   each MLP type; null heads are built from drawn q, k, v rows and o columns.
2. Mean removed ("centred" variant): each layer's mean vector (per type) is subtracted from its
   vectors, for the real layers and for the null.
3. Per ordered pair, variant (raw / centred) and outlier setting (all / no_outliers): fraction of
   neurons of i with a match in j against the covariance null of j (99th percentile of i's best
   similarities against the null, pooled over i's neurons, as in task 4b), fraction of mutual
   best matches between i and j, and the mutual fraction between i and the null of j. Same for
   heads (best match = largest head similarity; heads also matched one to one as in task 4b).
4. Within-distance Spearman with the task 4 loss increase, mean over distances 1..8, one-sided
   permutation p (1,000 shuffles within distance, seed 0), for the match fraction and the mutual
   fraction.

  python repeat/task4c.py --model main --seed 0
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
from repeat import task4b as B4  # noqa: E402
from repeat.task4 import MAX_DIST, out_dir, pairs, swap_matrix  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEV = B4.DEV
VARIANTS = ["raw", "centred"]
SETTINGS = B4.SETTINGS
TYPES = ["gate", "up", "down", "q", "k", "v", "o"]
ZCHUNK = 2048


def vectors(W, l):
    w = B4.fold(W.raw(l))
    return {"gate": w["gate"], "up": w["up"], "down": w["down"].T.contiguous(), "q": w["q"],
            "k": w["k"], "v": w["v"], "o": w["o"].T.contiguous()}


def cov_null(vecs, seed):
    """Same number of vectors per type, drawn from N(mean, sample covariance) of the layer."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for t in TYPES:
        X = vecs[t].double()
        mu = X.mean(0)
        Xc = ((X - mu) / np.sqrt(len(X) - 1)).float().to(DEV)
        n = len(X)
        rows = []
        for c0 in range(0, n, ZCHUNK):
            z = torch.randn((min(ZCHUNK, n - c0), n), generator=gen).to(DEV)
            rows.append((z @ Xc).cpu())
        out[t] = torch.cat(rows) + mu.float()
    return out


def centre(vecs):
    return {t: v - v.double().mean(0).float() for t, v in vecs.items()}


class Prepared:
    """Unit neuron vectors and head factors for both outlier settings, from a vector dict."""

    def __init__(self, W, vecs, outliers):
        H, KV, hd = W.H, W.KV, W.hd
        grp = (torch.arange(H) // (H // KV)).tolist()
        WO = torch.stack([vecs["o"][h * hd:(h + 1) * hd].T for h in range(H)])     # H x D x hd
        WV = torch.stack([vecs["v"][g * hd:(g + 1) * hd] for g in grp])           # H x hd x D
        WQt = torch.stack([vecs["q"][h * hd:(h + 1) * hd].T for h in range(H)])   # H x D x hd
        WK = torch.stack([vecs["k"][g * hd:(g + 1) * hd] for g in grp])           # H x hd x D
        self.neurons, self.heads = {}, {}
        for s in SETTINGS:
            keep = torch.ones(W.D)
            if s == "no_outliers":
                keep[outliers] = 0
            self.neurons[s] = tuple(B4.unit(vecs[t] * keep).to(DEV)
                                    for t in ("gate", "up", "down"))
            f = {"ov_u": WO * keep[None, :, None], "ov_v": WV * keep[None, None, :],
                 "qk_u": WQt * keep[None, :, None], "qk_v": WK * keep[None, None, :]}
            f = {k: v.to(DEV) for k, v in f.items()}
            for n in ("ov", "qk"):
                f[n + "_sq"] = B4.sqnorm(f[n + "_u"], f[n + "_v"])
            self.heads[s] = f


def layer_set(W, l, outliers, seed):
    v = vectors(W, l)
    nv = cov_null(v, seed * 1000 + l)
    out = {"raw": (Prepared(W, v, outliers), Prepared(W, nv, outliers)),
           "centred": (Prepared(W, centre(v), outliers), Prepared(W, centre(nv), outliers))}
    del v, nv
    return out


def mutual(rarg, carg):
    return float((carg[rarg] == np.arange(len(rarg))).mean())


def head_best(S):
    return S.max(1), S.argmax(1), S.max(0), S.argmax(0)


def compare(LA, LB, i, j):
    out = []
    for var in VARIANTS:
        A, An = LA[var]
        Bp, Bn = LB[var]
        for s in SETTINGS:
            rmax, rarg, cmax, carg = B4.neuron_sim(A.neurons[s], Bp.neurons[s])
            n_ij = B4.neuron_sim(A.neurons[s], Bn.neurons[s])     # i vs null of j
            n_ji = B4.neuron_sim(An.neurons[s], Bp.neurons[s])    # null of i vs j
            Sh = B4.head_sim(A.heads[s], Bp.heads[s])[0]
            Sn_ij = B4.head_sim(A.heads[s], Bn.heads[s])[0]
            Sn_ji = B4.head_sim(An.heads[s], Bp.heads[s])[0]
            hr, hra, hc, hca = head_best(Sh)
            _, _, mh = B4.assign(Sh)
            nr = head_best(Sn_ij)
            nc = head_best(Sn_ji)
            _, _, mhn_ij = B4.assign(Sn_ij)
            _, _, mhn_ji = B4.assign(Sn_ji)
            for a, b, best, mut, null_best, null_mut, hbest, hmut, hnull, hnull_mut, mhh, mhn in [
                    (i, j, rmax, mutual(rarg, carg), n_ij[0], mutual(n_ij[1], n_ij[3]),
                     hr, mutual(hra, hca), nr[0], mutual(nr[1], nr[3]), mh, mhn_ij),
                    (j, i, cmax, mutual(carg, rarg), n_ji[2], mutual(n_ji[3], n_ji[1]),
                     hc, mutual(hca, hra), nc[2], mutual(nc[3], nc[1]), mh, mhn_ji)]:
                thr = np.percentile(null_best, 99)
                hthr = np.percentile(hnull, 99)
                out.append({"i": a, "j": b, "dist": abs(i - j), "variant": var, "setting": s,
                            "neuron_match_frac": float((best > thr).mean()),
                            "neuron_mutual_frac": mut,
                            "neuron_mutual_frac_null": null_mut,
                            "neuron_median_best": float(np.median(best)),
                            "neuron_median_best_null": float(np.median(null_best)),
                            "neuron_null_p99": float(thr),
                            "head_match_frac": float((hbest > hthr).mean()),
                            "head_mutual_frac": hmut, "head_mutual_frac_null": hnull_mut,
                            "head_assigned_match_frac": float(
                                (mhh > np.percentile(mhn, 99)).mean()),
                            "head_mean_best": float(hbest.mean()),
                            "head_mean_best_null": float(np.mean(hnull))})
    return out


def link(rows, swap):
    res = {}
    rng = np.random.default_rng(0)
    for var in VARIANTS:
        for s in SETTINGS:
            rr = [r for r in rows if r["variant"] == var and r["setting"] == s
                  and (r["i"], r["j"]) in swap]
            y = np.array([swap[(r["i"], r["j"])] for r in rr])
            dist = np.array([r["dist"] for r in rr])
            for key in ["neuron_match_frac", "neuron_mutual_frac", "head_match_frac",
                        "head_mutual_frac"]:
                x = np.array([r[key] for r in rr])
                obs, per_d = B4_within(x, y, dist)
                perm = np.zeros(1000)
                for t in range(1000):
                    xs = x.copy()
                    for d in range(1, MAX_DIST + 1):
                        m = np.where(dist == d)[0]
                        xs[m] = xs[rng.permutation(m)]
                    perm[t] = B4_within(xs, y, dist)[0]
                res[f"{var}/{s}/{key}"] = {"mean_within_distance": obs, "within_distance": per_d,
                                           "perm_p_one_sided_negative": float((perm <= obs).mean()),
                                           "perm_p_two_sided": float(
                                               (np.abs(perm) >= abs(obs)).mean()),
                                           "n_pairs": len(rr)}
    return res


def B4_within(x, y, dist):
    from scipy.stats import spearmanr
    vals, per = [], {}
    for d in range(1, MAX_DIST + 1):
        m = dist == d
        rho = spearmanr(x[m], y[m])[0] if m.sum() > 2 and np.std(x[m]) > 0 else np.nan
        per[d] = float(rho)
        vals.append(rho)
    return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan"), per


def plots(rows, swap_rows, n, out):
    Hs = swap_matrix(swap_rows, n)
    for var in VARIANTS:
        for s in SETTINGS:
            fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
            mats = []
            for key in ["neuron_match_frac", "head_match_frac"]:
                M = np.full((n, n), np.nan)
                for r in rows:
                    if r["variant"] == var and r["setting"] == s:
                        M[r["i"], r["j"]] = r[key]
                mats.append(np.where(np.isnan(M), M.T, np.where(np.isnan(M.T), M,
                                                                 0.5 * (M + M.T))))
            for ax, M, title, cm in [(axs[0], np.log10(np.clip(Hs, 1e-3, None)),
                                      "task 4 swap loss increase, log10", "viridis_r"),
                                     (axs[1], mats[0], "neuron match fraction vs covariance null",
                                      "magma"),
                                     (axs[2], mats[1], "head match fraction vs covariance null",
                                      "magma")]:
                im = ax.imshow(M, origin="lower", cmap=cm)
                ax.set_title(f"{title} [{var}, {s}]", fontsize=9)
                ax.set_xlabel("j")
                ax.set_ylabel("i")
                fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"match_vs_swap_{var}_{s}.png"), dpi=110)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    out = out_dir(args.model, "task4c")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(out_dir(args.model, "task4b"), "summary.json")) as f:
        s4b = json.load(f)
    outliers = torch.tensor(s4b["outliers"])  # task 4b's outlier set, reused unchanged
    W = B4.Weights(E.MODELS[args.model])
    n = W.n_layers
    ordered = pairs(n)
    unordered = sorted({(min(i, j), max(i, j)) for i, j in ordered})
    by_a = {}
    for a, b in unordered:
        by_a.setdefault(a, []).append(b)
    rows, done = [], 0
    for a in sorted(by_a):
        LA = layer_set(W, a, outliers, args.seed)
        for b in by_a[a]:
            LB = layer_set(W, b, outliers, args.seed)
            rows += compare(LA, LB, a, b)
            del LB
            done += 1
            r = [x for x in rows[-8:] if x["i"] == a]
            print(f"pair {a},{b} ({done}/{len(unordered)}) {time.time() - t0:.0f}s " + " ".join(
                f"{x['variant'][0]}{x['setting'][0]}:{x['neuron_match_frac']:.3f}/"
                f"{x['neuron_mutual_frac']:.3f}/{x['neuron_mutual_frac_null']:.3f}" for x in r),
                flush=True)
        del LA
        torch.mps.empty_cache()
    oset = set(ordered)
    rows = [r for r in rows if (r["i"], r["j"]) in oset]
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir(args.model), "swap.csv")) as f:
        swap_rows = list(csv.DictReader(f))
    swap = {(int(r["i"]), int(r["j"])): float(r["loss_increase"]) for r in swap_rows}
    keys = ["neuron_match_frac", "neuron_mutual_frac", "neuron_mutual_frac_null",
            "neuron_median_best", "neuron_median_best_null", "head_match_frac",
            "head_mutual_frac", "head_mutual_frac_null", "head_assigned_match_frac"]
    summary = {"model": E.MODELS[args.model], "seed": args.seed, "outliers": s4b["outliers"],
               "n_ordered_pairs": len(ordered), "n_unordered_pairs": len(unordered),
               "by_distance": {f"{v}/{s}": {d: {k: float(np.mean(
                   [r[k] for r in rows if r["variant"] == v and r["setting"] == s
                    and r["dist"] == d])) for k in keys} for d in range(1, MAX_DIST + 1)}
                   for v in VARIANTS for s in SETTINGS},
               "link_to_task4": link(rows, swap), "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plots(rows, swap_rows, n, out)
    print(json.dumps(summary["link_to_task4"], indent=1)[:3000])


if __name__ == "__main__":
    main()
