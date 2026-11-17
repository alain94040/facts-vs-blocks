"""Repeat task 4b: do interchangeable layers share weights? (repeat/PLAN.md). Weights only.

1. Preparation: RMSNorm gains folded into the weights that read from them (q, k, v columns by
   input_layernorm; gate, up columns by post_attention_layernorm). Outlier set O: the 16 residual
   coordinates with the largest mean |x| at the input of the layer at 50% of depth, over the first
   20 TEXT SEARCH chunks (unmodified model). Every measurement is done with all coordinates
   ("all") and with the coordinates in O set to zero in every vector ("no_outliers"; vectors
   renormalised after zeroing for cosines).
2. MLP neurons: s(n, m) = cos(g_n, g_m) + |cos(u_n, u_m) + cos(d_n, d_m)|; best match and mutual
   best match.
3. Heads: OV_h = W_O,h W_V,g and QK_h = W_Q,h^T W_K,g (g = KV group of h), cosines computed from
   their 128-dim factors; head similarity = mean of OV and QK cosine; one-to-one matching with
   linear_sum_assignment. Biases are not part of these products; rotary embedding means QK is the
   pattern at relative offset 0 only.
4. Null: a shuffled copy of layer j, with an independent random permutation of the residual
   coordinates for every vector (each g, u, d row and each head factor matrix; seed 0, per layer).
   A neuron of i "has a match" in j if its best similarity in the real j is above the 99th
   percentile of the best similarities of i's neurons against the null copy of j (pooled over
   i's neurons). Same rule for matched heads (pooled over i's heads).
5. Positive control: the layer at 50% of depth, neurons and heads reordered (KV groups and the Q
   heads within each group), Gaussian noise at 30% of each weight matrix's standard deviation;
   at least 90% of neurons and heads must be recovered.
6. Pairs: the ordered pairs of task 4. Each unordered pair is computed once and gives both
   directions.
7.-9. Per ordered pair and setting: match fractions, medians, means, null levels; Spearman with
   the task 4 loss increase, overall and within each distance 1..8 (mean of the eight; one-sided
   permutation p for a negative mean, 1,000 shuffles within distance, seed 0); layer-by-layer
   matrices and histograms.

  python repeat/task4b.py --model main --seed 0
"""
import argparse
import csv
import glob
import json
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402
from repeat import flow as FL  # noqa: E402
from repeat.task4 import MAX_DIST, out_dir, pairs  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEV = torch.device("mps")
N_OUTLIER = 16
N_OUTLIER_CHUNKS = 20
CHUNK = 4096
NOISE = 0.3
SETTINGS = ["all", "no_outliers"]
HIST_DIST = [1, 4, 8]
BINS = np.linspace(-1, 3, 161)


class Weights:
    def __init__(self, name):
        d = glob.glob(os.path.join(ROOT, "data", "cache", "hf", "hub",
                                   "models--" + name.replace("/", "--"), "snapshots", "*"))[0]
        with open(os.path.join(d, "config.json")) as f:
            self.cfg = json.load(f)
        ix = os.path.join(d, "model.safetensors.index.json")
        if os.path.exists(ix):
            with open(ix) as f:
                wm = json.load(f)["weight_map"]
            self.files = {k: os.path.join(d, v) for k, v in wm.items()}
        else:
            f0 = os.path.join(d, "model.safetensors")
            with safe_open(f0, "pt") as f:
                self.files = {k: f0 for k in f.keys()}
        self.n_layers = self.cfg["num_hidden_layers"]
        self.H = self.cfg["num_attention_heads"]
        self.KV = self.cfg["num_key_value_heads"]
        self.D = self.cfg["hidden_size"]
        self.hd = self.D // self.H

    def raw(self, l):
        names = ["input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                 "mlp.down_proj.weight"]
        out = {}
        for n in names:
            key = f"model.layers.{l}.{n}"
            with safe_open(self.files[key], "pt") as f:
                out[n.split(".")[-2] if "proj" in n else n.split(".")[0]] = \
                    f.get_tensor(key).float()
        return out


def fold(w):
    """Fold the RMSNorm gains into the matrices that read from them."""
    g1, g2 = w["input_layernorm"], w["post_attention_layernorm"]
    return {"q": w["q_proj"] * g1, "k": w["k_proj"] * g1, "v": w["v_proj"] * g1,
            "o": w["o_proj"], "gate": w["gate_proj"] * g2, "up": w["up_proj"] * g2,
            "down": w["down_proj"]}


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def per_row_perm(x, gen):
    """Independent random permutation of the last dimension for every row."""
    idx = torch.argsort(torch.rand(x.shape, generator=gen), dim=-1)
    return torch.gather(x, -1, idx)


class Layer:
    """Prepared vectors of one layer for both settings, on DEV.
    neurons[s] = (G, U, Dn) unit rows (N x D); heads[s] = dict of factors (H x D x hd / H x hd x D)
    with their squared norms."""

    def __init__(self, W, l, outliers, shuffle_seed=None, raw=None):
        w = fold(raw if raw is not None else W.raw(l))
        H, KV, hd = W.H, W.KV, W.hd
        grp = torch.arange(H) // (H // KV)
        G, U, Dn = w["gate"], w["up"], w["down"].T.contiguous()  # rows = neurons
        WO = torch.stack([w["o"][:, h * hd:(h + 1) * hd] for h in range(H)])        # H x D x hd
        WV = torch.stack([w["v"][g * hd:(g + 1) * hd] for g in grp.tolist()])        # H x hd x D
        WQt = torch.stack([w["q"][h * hd:(h + 1) * hd].T for h in range(H)])         # H x D x hd
        WK = torch.stack([w["k"][g * hd:(g + 1) * hd] for g in grp.tolist()])        # H x hd x D
        if shuffle_seed is not None:
            gen = torch.Generator().manual_seed(shuffle_seed)
            G, U, Dn = (per_row_perm(t, gen) for t in (G, U, Dn))
            WO = per_row_perm(WO.transpose(1, 2), gen).transpose(1, 2).contiguous()
            WQt = per_row_perm(WQt.transpose(1, 2), gen).transpose(1, 2).contiguous()
            WV = per_row_perm(WV, gen)
            WK = per_row_perm(WK, gen)
        self.neurons, self.heads = {}, {}
        for s in SETTINGS:
            keep = torch.ones(W.D)
            if s == "no_outliers":
                keep[outliers] = 0
            self.neurons[s] = tuple(unit(t * keep).to(DEV) for t in (G, U, Dn))
            f = {"ov_u": (WO * keep[None, :, None]), "ov_v": (WV * keep[None, None, :]),
                 "qk_u": (WQt * keep[None, :, None]), "qk_v": (WK * keep[None, None, :])}
            f = {k: v.to(DEV) for k, v in f.items()}
            for n in ("ov", "qk"):
                f[n + "_sq"] = sqnorm(f[n + "_u"], f[n + "_v"])
            self.heads[s] = f


def sqnorm(u, v):
    # |U V|_F^2 = trace((U^T U)(V V^T)), per head
    return torch.einsum("hab,hba->h", u.transpose(1, 2) @ u, v @ v.transpose(1, 2))


def head_cos(fa, fb, n):
    """cos between product matrices of every head of a and every head of b: (H_a, H_b)."""
    P = torch.einsum("hda,kdb->hkab", fa[n + "_u"], fb[n + "_u"])     # U_A^T U_B
    Q = torch.einsum("kbd,had->hkba", fb[n + "_v"], fa[n + "_v"])     # V_B V_A^T
    ip = torch.einsum("hkab,hkba->hk", P, Q)
    return (ip / torch.sqrt(fa[n + "_sq"][:, None] * fb[n + "_sq"][None, :])).cpu().double()


def neuron_sim(na, nb):
    """Best match of every neuron of a in b (rows) and of b in a (columns)."""
    Ga, Ua, Da = na
    Gb, Ub, Db = nb
    N_a, N_b = Ga.shape[0], Gb.shape[0]
    rmax = torch.empty(N_a, device=DEV)
    rarg = torch.empty(N_a, dtype=torch.long, device=DEV)
    cmax = torch.full((N_b,), -10.0, device=DEV)
    carg = torch.zeros(N_b, dtype=torch.long, device=DEV)
    for c0 in range(0, N_a, CHUNK):
        sl = slice(c0, min(N_a, c0 + CHUNK))
        S = Ga[sl] @ Gb.T + (Ua[sl] @ Ub.T + Da[sl] @ Db.T).abs()
        rmax[sl], rarg[sl] = S.max(1)
        m, a = S.max(0)
        better = m > cmax
        cmax = torch.where(better, m, cmax)
        carg = torch.where(better, a + c0, carg)
    return rmax.cpu().numpy(), rarg.cpu().numpy(), cmax.cpu().numpy(), carg.cpu().numpy()


def head_sim(fa, fb):
    return 0.5 * (head_cos(fa, fb, "ov") + head_cos(fa, fb, "qk")).numpy(), \
        head_cos(fa, fb, "ov").numpy(), head_cos(fa, fb, "qk").numpy()


def assign(S):
    r, c = linear_sum_assignment(-S)
    return r, c, S[r, c]


# ---------------------------------------------------------------- outliers

def outlier_set(model_key, layer):
    m, tok = FL.load(E.MODELS[model_key])
    flow = FL.Flow(m)
    sets = E.build_sets(tok, E.K_FROZEN)
    idx = sets["splits"]["TEXT"]["SEARCH"][:N_OUTLIER_CHUNKS]
    acc = None
    for i in idx:
        x = torch.tensor([sets["TEXT"][i]]).to(DEV)
        h = flow.hidden(x, stop_at=layer)[0].float().abs().mean(0).cpu()
        acc = h if acc is None else acc + h
    acc = acc / len(idx)
    top = torch.argsort(acc, descending=True)[:N_OUTLIER]
    del m, flow
    torch.mps.empty_cache()
    return top, acc


# ---------------------------------------------------------------- positive control

def positive_control(W, l, outliers, seed):
    raw = W.raw(l)
    gen = torch.Generator().manual_seed(seed)
    H, KV, hd = W.H, W.KV, W.hd
    per = H // KV
    N = raw["gate_proj"].shape[0]
    pn = torch.randperm(N, generator=gen)                      # new neuron k = old neuron pn[k]
    pg = torch.randperm(KV, generator=gen)                     # new group g = old group pg[g]
    within = [torch.randperm(per, generator=gen) for _ in range(KV)]
    ph = torch.cat([pg[g] * per + within[g] for g in range(KV)])  # new head k = old head ph[k]
    rows = lambda w, perm, size: torch.cat([w[p * size:(p + 1) * size] for p in perm.tolist()])  # noqa: E731
    c = dict(raw)
    c["gate_proj"], c["up_proj"] = raw["gate_proj"][pn], raw["up_proj"][pn]
    c["down_proj"] = raw["down_proj"][:, pn]
    c["q_proj"] = rows(raw["q_proj"], ph, hd)
    c["k_proj"] = rows(raw["k_proj"], pg, hd)
    c["v_proj"] = rows(raw["v_proj"], pg, hd)
    c["o_proj"] = torch.cat([raw["o_proj"][:, p * hd:(p + 1) * hd] for p in ph.tolist()], 1)
    for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        c[k] = c[k] + NOISE * c[k].std() * torch.randn(c[k].shape, generator=gen)
    A = Layer(W, l, outliers, raw=raw)
    B = Layer(W, l, outliers, raw=c)
    true_n = torch.argsort(pn).numpy()   # old neuron n sits at new index true_n[n]
    true_h = torch.argsort(ph).numpy()
    res = {}
    for s in SETTINGS:
        rmax, rarg, _, _ = neuron_sim(A.neurons[s], B.neurons[s])
        Sh, _, _ = head_sim(A.heads[s], B.heads[s])
        r, cc, _ = assign(Sh)
        res[s] = {"neuron_recovered": float((rarg == true_n).mean()),
                  "head_recovered": float((cc[np.argsort(r)] == true_h).mean()),
                  "median_neuron_similarity": float(np.median(rmax))}
    res["layer"] = l
    res["noise"] = NOISE
    res["pass"] = all(res[s]["neuron_recovered"] >= 0.9 and res[s]["head_recovered"] >= 0.9
                      for s in SETTINGS)
    del A, B
    torch.mps.empty_cache()
    return res


# ---------------------------------------------------------------- pairs

def compare_pair(A, B, Anull, Bnull, i, j, hist):
    """Rows for (i -> j) and (j -> i), both settings."""
    out = []
    d = abs(i - j)
    for s in SETTINGS:
        rmax, rarg, cmax, carg = neuron_sim(A.neurons[s], B.neurons[s])
        mutual_ij = float((carg[rarg] == np.arange(len(rarg))).mean())
        mutual_ji = float((rarg[carg] == np.arange(len(carg))).mean())
        null_ij = neuron_sim(A.neurons[s], Bnull.neurons[s])[0]        # i vs shuffled j
        null_ji = neuron_sim(Anull.neurons[s], B.neurons[s])[2]        # j vs shuffled i
        Sh, Sov, Sqk = head_sim(A.heads[s], B.heads[s])
        r, c, mh = assign(Sh)
        Shn_ij, Sovn_ij, Sqkn_ij = head_sim(A.heads[s], Bnull.heads[s])
        rn, cn, mhn_ij = assign(Shn_ij)
        Shn_ji, Sovn_ji, Sqkn_ji = head_sim(Anull.heads[s], B.heads[s])
        rn2, cn2, mhn_ji = assign(Shn_ji)
        for (a, b, best, mut, null, hnull, ovn, qkn) in [
                (i, j, rmax, mutual_ij, null_ij, mhn_ij, Sovn_ij[rn, cn], Sqkn_ij[rn, cn]),
                (j, i, cmax, mutual_ji, null_ji, mhn_ji, Sovn_ji[rn2, cn2], Sqkn_ji[rn2, cn2])]:
            thr = np.percentile(null, 99)
            hthr = np.percentile(hnull, 99)
            out.append({"i": a, "j": b, "dist": d, "setting": s,
                        "neuron_match_frac": float((best > thr).mean()),
                        "neuron_mutual_frac": mut,
                        "neuron_median_best": float(np.median(best)),
                        "neuron_median_best_null": float(np.median(null)),
                        "neuron_null_p99": float(thr),
                        "head_match_frac": float((mh > hthr).mean()),
                        "head_mean_matched_sim": float(mh.mean()),
                        "head_mean_ov_cos": float(Sov[r, c].mean()),
                        "head_mean_qk_cos": float(Sqk[r, c].mean()),
                        "head_mean_ov_cos_null": float(ovn.mean()),
                        "head_mean_qk_cos_null": float(qkn.mean()),
                        "head_null_p99": float(hthr)})
            if d in HIST_DIST:
                hist[s][d]["real"] += np.histogram(best, BINS)[0]
                hist[s][d]["null"] += np.histogram(null, BINS)[0]
    return out


def link(rows, swap):
    """Spearman between match fraction and swap loss increase (ordered pairs)."""
    res = {}
    rng = np.random.default_rng(0)
    for s in SETTINGS:
        for key in ["neuron_match_frac", "head_match_frac"]:
            rr = [r for r in rows if r["setting"] == s and (r["i"], r["j"]) in swap]
            x = np.array([r[key] for r in rr])
            y = np.array([swap[(r["i"], r["j"])] for r in rr])
            dist = np.array([r["dist"] for r in rr])
            overall = spearmanr(x, y)
            within, per_d = [], {}
            for d in range(1, MAX_DIST + 1):
                m = dist == d
                rho = spearmanr(x[m], y[m])[0] if m.sum() > 2 and np.std(x[m]) > 0 else np.nan
                per_d[d] = float(rho)
                within.append(rho)
            obs = float(np.nanmean(within))
            perm = np.zeros(1000)
            for t in range(1000):
                xs = x.copy()
                for d in range(1, MAX_DIST + 1):
                    m = np.where(dist == d)[0]
                    xs[m] = xs[rng.permutation(m)]
                vals = [spearmanr(xs[dist == d], y[dist == d])[0]
                        if (dist == d).sum() > 2 and np.std(xs[dist == d]) > 0 else np.nan
                        for d in range(1, MAX_DIST + 1)]
                perm[t] = np.nanmean(vals)
            res[f"{s}/{key}"] = {
                "n_pairs": len(rr), "spearman_overall": float(overall[0]),
                "spearman_overall_p": float(overall[1]), "within_distance": per_d,
                "mean_within_distance": obs,
                "perm_p_one_sided_negative": float((perm <= obs).mean()),
                "perm_p_two_sided": float((np.abs(perm) >= abs(obs)).mean())}
    return res


def plots(rows, swap_rows, n, out, hist):
    from repeat.task4 import swap_matrix
    Hs = swap_matrix(swap_rows, n)
    for s in SETTINGS:
        Mn = np.full((n, n), np.nan)
        Mh = np.full((n, n), np.nan)
        for r in rows:
            if r["setting"] == s:
                Mn[r["i"], r["j"]] = r["neuron_match_frac"]
                Mh[r["i"], r["j"]] = r["head_match_frac"]
        sym = lambda M: np.where(np.isnan(M) | np.isnan(M.T), np.where(np.isnan(M), M.T, M),  # noqa: E731
                                 0.5 * (M + M.T))
        fig, axs = plt.subplots(1, 3, figsize=(19, 5.5))
        for ax, M, title, cm in [(axs[0], np.log10(np.clip(Hs, 1e-3, None)),
                                  "task 4 swap(i, j) TEXT loss increase, log10", "viridis_r"),
                                 (axs[1], sym(Mn), "neuron match fraction (mean of directions)",
                                  "magma"),
                                 (axs[2], sym(Mh), "head match fraction (mean of directions)",
                                  "magma")]:
            im = ax.imshow(M, origin="lower", cmap=cm)
            ax.set_title(f"{title} [{s}]", fontsize=9)
            ax.set_xlabel("j")
            ax.set_ylabel("i")
            fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"match_vs_swap_{s}.png"), dpi=110)
        plt.close(fig)
    fig, axs = plt.subplots(len(SETTINGS), len(HIST_DIST), figsize=(15, 7), squeeze=False)
    c = 0.5 * (BINS[1:] + BINS[:-1])
    for a, s in enumerate(SETTINGS):
        for b, d in enumerate(HIST_DIST):
            ax = axs[a, b]
            for k, col in [("real", "C0"), ("null", "0.5")]:
                h = hist[s][d][k]
                if h.sum():
                    ax.step(c, h / h.sum(), color=col, label=k)
            ax.set_title(f"|i-j| = {d}, {s}", fontsize=9)
            ax.set_xlabel("best-match similarity s")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "best_match_hist.png"), dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control_only", action="store_true")
    args = p.parse_args()
    t0 = time.time()
    out = out_dir(args.model, "task4b")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    W = Weights(E.MODELS[args.model])
    n = W.n_layers
    mid = round(0.5 * n)
    outliers, mag = outlier_set(args.model, mid)
    summary = {"model": E.MODELS[args.model], "seed": args.seed, "outlier_layer": mid,
               "outliers": outliers.tolist(),
               "outlier_mean_abs": mag[outliers].tolist(),
               "median_mean_abs": float(mag.median())}
    print("outliers", summary["outliers"], flush=True)
    pc = positive_control(W, mid, outliers, args.seed)
    summary["positive_control"] = pc
    print("positive control", pc, round(time.time() - t0), "s", flush=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if not pc["pass"]:
        with open(os.path.join(out, "STOP.md"), "w") as f:
            f.write("# STOP: Repeat task 4b positive control failed\n\n" + json.dumps(pc, indent=2)
                    + "\n")
        print("POSITIVE CONTROL FAILED", flush=True)
        sys.exit(1)
    if args.control_only:
        return

    ordered = pairs(n)
    unordered = sorted({(min(i, j), max(i, j)) for i, j in ordered})
    hist = {s: {d: {"real": np.zeros(len(BINS) - 1), "null": np.zeros(len(BINS) - 1)}
                for d in HIST_DIST} for s in SETTINGS}
    rows = []
    by_a = {}
    for a, b in unordered:
        by_a.setdefault(a, []).append(b)
    done = 0
    for a in sorted(by_a):
        A = Layer(W, a, outliers)
        Anull = Layer(W, a, outliers, shuffle_seed=args.seed * 1000 + a)
        for b in by_a[a]:
            B = Layer(W, b, outliers)
            Bnull = Layer(W, b, outliers, shuffle_seed=args.seed * 1000 + b)
            rows += compare_pair(A, B, Anull, Bnull, a, b, hist)
            del B, Bnull
            done += 1
            print(f"pair {a},{b} ({done}/{len(unordered)}) {time.time() - t0:.0f}s "
                  + " ".join(f"{r['setting']}:{r['neuron_match_frac']:.3f}/{r['head_match_frac']:.2f}"
                             for r in rows[-4:] if r["i"] == a), flush=True)
        del A, Anull
        torch.mps.empty_cache()
    ordered_set = set(ordered)
    rows = [r for r in rows if (r["i"], r["j"]) in ordered_set]
    with open(os.path.join(out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir(args.model), "swap.csv")) as f:
        swap_rows = list(csv.DictReader(f))
    swap = {(int(r["i"]), int(r["j"])): float(r["loss_increase"]) for r in swap_rows}
    summary["n_ordered_pairs"] = len(ordered)
    summary["n_unordered_pairs"] = len(unordered)
    summary["link_to_task4"] = link(rows, swap)
    summary["by_distance"] = {
        s: {d: {k: float(np.mean([r[k] for r in rows if r["setting"] == s and r["dist"] == d]))
                for k in ["neuron_match_frac", "neuron_mutual_frac", "head_match_frac",
                          "neuron_median_best", "neuron_median_best_null"]}
            for d in range(1, MAX_DIST + 1)} for s in SETTINGS}
    summary["wall_seconds"] = time.time() - t0
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plots(rows, swap_rows, n, out, hist)
    print(json.dumps({k: v for k, v in summary.items() if k != "by_distance"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
