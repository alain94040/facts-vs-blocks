"""Task 5: do the same components recur (PLAN.md Phase 2, with the answers in QUESTIONS.md).

Head features, per head (6 x 8) of every model:
  STORY attention scores (all models): story_induction, story_duplicate, prev_token, sink.
  FORMAT attention scores (biography models only): format_induction, format_duplicate.
  Weight scores (all models): copying (sum Re(eig) / sum |eig| of W_OV W_E^T W_E; W_E over the full
  vocabulary = copying_full, over the 4,096 base tokens = copying_base), and spectral signatures:
  top 16 singular values of W_Q W_K^T and of W_V W_O, each divided by its largest.
  Ablation effects (single-head mean-ablation, Task 3 step 5): d_recall, d_format_loss,
  d_name_copy, d_story_loss for biography models (read from Task 3); d_story_loss only for
  main-0 (computed here, reference = 1,000 STORY rows); zero for the untrained model.
Feature sets:
  FULL  = 4 STORY scores + 2 FORMAT scores + copying_full + 32 spectral + 4 ablation effects.
  STORY = 4 STORY scores + copying_base + 32 spectral + d_story_loss.
Standardisation over all heads of all trained models that have the feature set (columns with zero
spread are dropped).

Head matching: nearest head (Euclidean) in seed 1 for each seed-0 head; per-head null from 100
independent column shuffles of the seed-1 heads; recurs if below the null's 5th percentile.
Neuron matching: best Pearson correlation (max over the same layer of the other model) on 200,000
fixed positions (all 78,047 FORMAT positions + 121,953 STORY, or 200k STORY when main-0 is
involved); null = seed 0
against the untrained model on the same positions.

  python analysis/task5.py --out results/task5 --task3 results/task3 --seed 0
  python analysis/task5.py --pilot --out results/task5_pilot --task3 results/task3_pilot
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
from analysis import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

UNTRAINED = "untrained-s0"
N_SPEC = 16
N_SHUFFLE = 100
N_POS = 200_000
N_CLUSTERS = 8
STORY_SCORES = ["story_induction", "story_duplicate", "prev_token", "sink"]
FORMAT_SCORES = ["format_induction", "format_duplicate"]
SPEC = [f"qk_sv{i}" for i in range(N_SPEC)] + [f"ov_sv{i}" for i in range(N_SPEC)]
ABL = ["d_recall", "d_format_loss", "d_name_copy", "d_story_loss"]
FEATURES = {
    "FULL": STORY_SCORES + FORMAT_SCORES + ["copying_full"] + SPEC + ABL,
    "STORY": STORY_SCORES + ["copying_base"] + SPEC + ["d_story_loss"],
}

REAL = {
    "bio": ["main-2k-s0", "main-2k-s1", "main-8k-s0", "main-8k-s1", "main-32k-s0", "main-32k-s1",
            "ctrl-8k-x4-s0", "fork-A-s0", "fork-B-s0"],
    "nobio": ["main-0-s0", "main-0-s1"],
    "head_pairs": [("main-0", "main-0-s0", "main-0-s1", "STORY"),
                   ("main-2k", "main-2k-s0", "main-2k-s1", "FULL"),
                   ("main-8k", "main-8k-s0", "main-8k-s1", "FULL"),
                   ("main-32k", "main-32k-s0", "main-32k-s1", "FULL"),
                   ("main-2k-s0 vs main-32k-s0", "main-2k-s0", "main-32k-s0", "FULL")],
    "neuron_pairs": [("main-0", "main-0-s0", "main-0-s1"), ("main-2k", "main-2k-s0", "main-2k-s1"),
                     ("main-8k", "main-8k-s0", "main-8k-s1"),
                     ("main-32k", "main-32k-s0", "main-32k-s1")],
}
# Development only: two unrelated pilot models stand in for a seed pair; P1 also exercises the
# no-biography code path.
PILOT = {
    "bio": ["P2-main-2k-s0"],
    "nobio": ["P1-main-2k-s0"],
    "head_pairs": [("pilot-story", "P2-main-2k-s0", "P1-main-2k-s0", "STORY")],
    "neuron_pairs": [("pilot", "P2-main-2k-s0", "P1-main-2k-s0")],
}


def load(run):
    return C.untrained_model() if run == UNTRAINED else C.load_model(run)


# ---------------------------------------------------------------- attention scores

def attention_probs(blk, x_ln):
    """Attention probabilities (B, H, T, T) of one block, from the ln1 output."""
    B, T, D = x_ln.shape
    q, k, _ = blk.attn.qkv(x_ln).split(D, dim=2)
    q = q.view(B, T, C.N_HEAD, C.D_HEAD).transpose(1, 2)
    k = k.view(B, T, C.N_HEAD, C.D_HEAD).transpose(1, 2)
    s = (q @ k.transpose(-1, -2)) / np.sqrt(C.D_HEAD)
    causal = torch.ones(T, T, dtype=torch.bool, device=x_ln.device).tril()
    return s.masked_fill(~causal, float("-inf")).softmax(-1)


@torch.no_grad()
def per_layer_attention(model, x):
    """Yield (layer, attention probs) for a batch, running the model's own forward."""
    pos = torch.arange(x.shape[1], device=x.device)
    h = model.tok_emb(x) + model.pos_emb(pos)
    for l, blk in enumerate(model.blocks):
        yield l, attention_probs(blk, blk.ln1(h))
        h = blk(h)


@torch.no_grad()
def story_scores(model):
    rows = C.story_rows()[:, :-1]
    acc = {k: torch.zeros(C.N_LAYER, C.N_HEAD, dtype=torch.float64) for k in STORY_SCORES}
    n_rep, n_pos = 0, 0
    for i in range(0, len(rows), 32):
        x = torch.from_numpy(rows[i:i + 32]).to(C.DEVICE)
        B, T = x.shape
        same = (x[:, :, None] == x[:, None, :]) & torch.ones(T, T, dtype=torch.bool,
                                                             device=x.device).tril(-1)
        nxt = torch.zeros_like(same)
        nxt[:, :, 1:] = same[:, :, :-1]  # position j+1 for each earlier occurrence j
        rep = same.any(-1)  # positions whose token occurred earlier in the row
        n_rep += int(rep.sum())
        n_pos += B * (T - 1)
        for l, A in per_layer_attention(model, x):
            Af = A.float()
            acc["story_duplicate"][l] += ((Af * same[:, None]).sum(-1) * rep[:, None]).sum(
                (0, 2)).cpu().double()
            acc["story_induction"][l] += ((Af * nxt[:, None]).sum(-1) * rep[:, None]).sum(
                (0, 2)).cpu().double()
            idx = torch.arange(1, T, device=x.device)
            acc["prev_token"][l] += Af[:, :, idx, idx - 1].sum((0, 2)).cpu().double()
            acc["sink"][l] += Af[:, :, 1:, 0].sum((0, 2)).cpu().double()
    out = {}
    for k in ["story_duplicate", "story_induction"]:
        out[k] = (acc[k] / n_rep).numpy()
    for k in ["prev_token", "sink"]:
        out[k] = (acc[k] / n_pos).numpy()
    return out


@torch.no_grad()
def format_scores(model):
    """At repeated mentions of the name in FORMAT biographies: attention mass from the bare-space
    position onto earlier last-name tokens (induction), and from the first-name position onto
    earlier first-name tokens (duplicate)."""
    tok = C.ev.tokenizer()
    vocab = tok.get_vocab()
    space = tok.encode(" ").ids[0]
    bios = C.format_bios()
    with open(os.path.join(C.ROOT, "data", "cache", "bios", "people_B.json")) as f:
        people = json.load(f)[C.ev.NAME_COPY_PEOPLE[0]:C.ev.NAME_COPY_PEOPLE[1]]
    ind = torch.zeros(C.N_LAYER, C.N_HEAD, dtype=torch.float64)
    dup = torch.zeros(C.N_LAYER, C.N_HEAD, dtype=torch.float64)
    n = 0
    for b0 in range(0, len(bios), 125):
        chunk = bios[b0:b0 + 125]
        L = max(len(ids) for ids, _ in chunk)
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        checks = []  # (row, first-name position, earlier first-name positions)
        for r, (ids, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
            first, last = people[b0 + r]["name"].split(" ")
            fid, lid = vocab[first], vocab[last]
            pos = [i for i in range(len(ids) - 2)
                   if ids[i] == fid and ids[i + 1] == space and ids[i + 2] == lid]
            assert len(pos) == len(C.ATTRIBUTES), len(pos)
            for m in range(1, len(pos)):
                checks.append((r, pos[m], pos[:m]))
        x = x.to(C.DEVICE)
        for l, A in per_layer_attention(model, x):
            Ac = A.float().cpu()
            for r, i, earlier in checks:
                e = torch.tensor(earlier)
                ind[l] += Ac[r, :, i + 1, e + 2].sum(-1).double()
                dup[l] += Ac[r, :, i, e].sum(-1).double()
        n += len(checks)
    return {"format_induction": (ind / n).numpy(), "format_duplicate": (dup / n).numpy()}


# ---------------------------------------------------------------- weight scores

def weight_scores(model):
    E = model.tok_emb.weight.detach().cpu().double()
    gram = {"copying_full": E.T @ E, "copying_base": E[:C.ev.BASE_VOCAB].T @ E[:C.ev.BASE_VOCAB]}
    out = {k: np.zeros((C.N_LAYER, C.N_HEAD)) for k in ["copying_full", "copying_base"]}
    for s in SPEC:
        out[s] = np.zeros((C.N_LAYER, C.N_HEAD))
    D = C.N_HEAD * C.D_HEAD
    for l, blk in enumerate(model.blocks):
        W = blk.attn.qkv.weight.detach().cpu().double()  # (3D, D): q = W[:D] x
        Wo = blk.attn.out.weight.detach().cpu().double()  # (D, D): out = Wo y
        for h in range(C.N_HEAD):
            sl = slice(h * C.D_HEAD, (h + 1) * C.D_HEAD)
            Wq, Wk, Wv = W[:D][sl], W[D:2 * D][sl], W[2 * D:][sl]  # (32, D) each
            ov = Wo[:, sl] @ Wv  # (D, D) maps a residual vector to the head's output
            qk = Wq.T @ Wk  # (D, D) bilinear form x_q^T qk x_k
            for k, G in gram.items():
                ev = np.linalg.eigvals((ov @ G).numpy())
                out[k][l, h] = ev.real.sum() / np.abs(ev).sum()
            for name, M in [("qk", qk), ("ov", ov)]:
                sv = np.linalg.svd(M.numpy(), compute_uv=False)[:N_SPEC]
                for i in range(N_SPEC):
                    out[f"{name}_sv{i}"][l, h] = sv[i] / sv[0]
    return out


def ablation_scores(run, model, bio, task3_dir):
    z = np.zeros((C.N_LAYER, C.N_HEAD))
    out = {k: z.copy() for k in ABL}
    if run == UNTRAINED:
        return out
    if bio:
        with open(os.path.join(task3_dir, run, "metrics.json")) as f:
            heads = json.load(f)["heads"]
        for h in heads:
            for k in ABL:
                out[k][h["layer"], h["head"]] = h[k]
        return out
    means, _ = C.component_means(model, story_only=True)
    meta = C.run_meta(run)
    base = C.mean_loss(model, "STORY", meta)
    for l in range(C.N_LAYER):
        for h in range(C.N_HEAD):
            with C.Hooks(model, means, ablate_heads={l: [h]}):
                out["d_story_loss"][l, h] = C.mean_loss(model, "STORY", meta) - base
    for k in ["d_recall", "d_format_loss", "d_name_copy"]:
        out[k][:] = np.nan
    return out


def head_features(run, bio, task3_dir):
    model = load(run)
    f = {**story_scores(model), **weight_scores(model), **ablation_scores(run, model, bio,
                                                                          task3_dir)}
    if bio:
        f.update(format_scores(model))
    else:
        for k in FORMAT_SCORES:
            f[k] = np.full((C.N_LAYER, C.N_HEAD), np.nan)
    rows = []
    for l in range(C.N_LAYER):
        for h in range(C.N_HEAD):
            rows.append({"model": run, "layer": l, "head": h, "bio": bio,
                         **{k: float(v[l, h]) for k, v in f.items()}})
    return rows


# ---------------------------------------------------------------- head matching

def standardiser(rows, feats):
    X = np.array([[r[k] for k in feats] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-12
    return lambda R: (np.array([[r[k] for k in feats] for r in R])[:, keep] - mu[keep]) / sd[keep], \
        [f for f, k in zip(feats, keep) if k]


def nearest(X0, X1):
    d = np.sqrt(((X0[:, None, :] - X1[None, :, :]) ** 2).sum(-1))
    return d.min(1), d.argmin(1)


def match_heads(Z0, Z1, rng):
    d, j = nearest(Z0, Z1)
    null = np.zeros((N_SHUFFLE, len(Z0)))
    for s in range(N_SHUFFLE):
        Zs = np.stack([rng.permutation(Z1[:, c]) for c in range(Z1.shape[1])], 1)
        null[s] = nearest(Z0, Zs)[0]
    thr = np.percentile(null, 5, axis=0)
    rec = d < thr
    return {"fraction_recur": float(rec.mean()), "n_heads": len(Z0),
            "distance": d.tolist(), "null_5th_percentile": thr.tolist(),
            "null_median": np.median(null, 0).tolist(), "recurs": rec.tolist(),
            "match": [int(x) for x in j],
            "match_same_layer": float(np.mean(np.array(j) // C.N_HEAD
                                              == np.arange(len(j)) // C.N_HEAD))}


def kmeans(Z, k, rng, n_init=10, iters=300):
    best = None
    for _ in range(n_init):
        cent = Z[rng.choice(len(Z), 1)]
        while len(cent) < k:  # k-means++ seeding
            d2 = ((Z[:, None] - cent[None]) ** 2).sum(-1).min(1)
            cent = np.vstack([cent, Z[rng.choice(len(Z), 1, p=d2 / d2.sum())]])
        for _ in range(iters):
            lab = ((Z[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
            new = np.array([Z[lab == c].mean(0) if (lab == c).any() else cent[c]
                            for c in range(k)])
            if np.allclose(new, cent):
                break
            cent = new
        inertia = ((Z - cent[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, cent)
    return best


def describe_clusters(rows, Z, feats, rng):
    inertia, lab, _ = kmeans(Z, N_CLUSTERS, rng)
    out = []
    for c in range(N_CLUSTERS):
        members = [r for r, l in zip(rows, lab) if l == c]
        models = sorted({r["model"] for r in members})
        out.append({"cluster": c, "n_heads": len(members), "n_models": len(models),
                    "models": models,
                    "mean_features": {k: float(np.mean([r[k] for r in members])) for k in feats},
                    "heads": [f"{r['model']}:L{r['layer']}H{r['head']}" for r in members]})
    return {"inertia": float(inertia), "clusters": out}


# ---------------------------------------------------------------- neuron matching

def positions(mixed, rng):
    """Fixed positions: (story (row, pos) array, format (bio, pos) array). Mixed = every FORMAT
    position (78,047) plus random STORY positions up to 200,000 (QUESTIONS.md, Task 5 answer 4)."""
    fmt = np.zeros((0, 2), dtype=np.int64)
    if mixed:
        fmt = np.array([(r, i) for r, (ids, _) in enumerate(C.format_bios())
                        for i in range(len(ids))])
    n_story = N_POS - len(fmt)
    T = C.story_rows().shape[1] - 1
    flat = rng.choice(len(C.story_rows()) * T, n_story, replace=False)
    story = np.stack([flat // T, flat % T], 1)
    return story, fmt


@torch.no_grad()
def neuron_acts(model, story, fmt):
    """Post-GELU activations at the positions, per layer: list of (n, 1024) float32 arrays."""
    acts = [[] for _ in range(C.N_LAYER)]
    with C.Hooks(model, capture=True) as hk:
        rows = C.story_rows()[:, :-1]
        for i in range(0, len(rows), 128):
            sel = story[(story[:, 0] >= i) & (story[:, 0] < i + 128)]
            model(torch.from_numpy(rows[i:i + 128]).to(C.DEVICE))
            for l in range(C.N_LAYER):
                a = hk.acts[("mlp", l)]
                acts[l].append(a[torch.from_numpy(sel[:, 0] - i), torch.from_numpy(sel[:, 1])]
                               .cpu().numpy())
        bios = C.format_bios()
        for i in range(0, len(bios), 250):
            sel = fmt[(fmt[:, 0] >= i) & (fmt[:, 0] < i + 250)]
            if not len(sel):
                continue
            chunk = bios[i:i + 250]
            x = torch.zeros((len(chunk), max(len(ids) for ids, _ in chunk)), dtype=torch.long)
            for r, (ids, _) in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids)
            model(x.to(C.DEVICE))
            for l in range(C.N_LAYER):
                a = hk.acts[("mlp", l)]
                acts[l].append(a[torch.from_numpy(sel[:, 0] - i), torch.from_numpy(sel[:, 1])]
                               .cpu().numpy())
    return [np.concatenate(a) for a in acts]


def best_corr(A0, A1):
    """For each neuron of A0, the max Pearson correlation with any neuron of A1 (float64, CPU).
    Constant neurons get correlation 0."""
    def z(A):
        A = A.astype(np.float64)
        A = A - A.mean(0)
        sd = np.sqrt((A ** 2).mean(0))
        return np.where(sd > 0, A / np.where(sd > 0, sd, 1), 0.0)
    Cm = z(A0).T @ z(A1) / len(A0)
    return Cm.max(1), Cm.argmax(1)


def knowledge_set(task3_dir, run):
    p = os.path.join(task3_dir, run, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        K = json.load(f)["knowledge_set"]["K"]
    if not K:
        return None
    imp = np.load(os.path.join(task3_dir, run, "importance.npz"))["neurons_FACT"].ravel()
    return set(np.argsort(-imp, kind="stable")[:K].tolist())


def neuron_matching(cfg, task3_dir, seed):
    out = {}
    cache = {}
    for name, r0, r1 in cfg["neuron_pairs"]:
        mixed = r0 not in cfg["nobio"]
        key = "mixed" if mixed else "story"
        if key not in cache:
            cache[key] = positions(mixed, np.random.default_rng(seed))
        story, fmt = cache[key]
        a0 = neuron_acts(load(r0), story, fmt)
        ks = knowledge_set(task3_dir, r0)
        res = {"seed0": r0, "positions": {"story": len(story), "format": len(fmt)}}
        for label, other in [("seed1", r1), ("null_untrained", UNTRAINED)]:
            a1 = neuron_acts(load(other), story, fmt)
            per_layer, allc = [], []
            for l in range(C.N_LAYER):
                bc, _ = best_corr(a0[l], a1[l])
                allc.append(bc)
                per_layer.append({"layer": l, "frac_above_0.5": float((bc > 0.5).mean()),
                                  "frac_above_0.8": float((bc > 0.8).mean()),
                                  "median_best_corr": float(np.median(bc))})
            allc = np.concatenate(allc)
            d = {"other": other, "by_layer": per_layer,
                 "frac_above_0.5": float((allc > 0.5).mean()),
                 "frac_above_0.8": float((allc > 0.8).mean())}
            if ks is not None:
                inside = np.zeros(len(allc), dtype=bool)
                inside[list(ks)] = True
                for part, m in [("inside_knowledge_set", inside), ("outside_knowledge_set", ~inside)]:
                    d[part] = {"n": int(m.sum()), "frac_above_0.5": float((allc[m] > 0.5).mean()),
                               "frac_above_0.8": float((allc[m] > 0.8).mean())}
            res[label] = d
            print("neurons", name, label, d["frac_above_0.5"], d["frac_above_0.8"], flush=True)
        out[name] = res
    return out


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/task5")
    p.add_argument("--task3", default="results/task3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pilot", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    cfg = PILOT if args.pilot else REAL
    out = os.path.join(C.ROOT, args.out)
    task3 = os.path.join(C.ROOT, args.task3)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")

    rows = []
    for run in cfg["bio"] + cfg["nobio"] + [UNTRAINED]:
        rows += head_features(run, run in cfg["bio"], task3)
        print("features", run, round(time.time() - t0), "s", flush=True)
    cols = list(rows[0].keys())
    with open(os.path.join(out, "heads.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    trained = cfg["bio"] + cfg["nobio"]
    std = {"FULL": standardiser([r for r in rows if r["model"] in cfg["bio"]], FEATURES["FULL"]),
           "STORY": standardiser([r for r in rows if r["model"] in trained], FEATURES["STORY"])}
    rng = np.random.default_rng(args.seed)
    hm = {"feature_sets": {k: v[1] for k, v in std.items()}, "pairs": {}}
    for name, r0, r1, fs in cfg["head_pairs"]:
        tf = std[fs][0]
        res = match_heads(tf([r for r in rows if r["model"] == r0]),
                          tf([r for r in rows if r["model"] == r1]), rng)
        hm["pairs"][name] = {"seed0": r0, "seed1": r1, "feature_set": fs, **res}
        print("heads", name, fs, res["fraction_recur"], flush=True)
    bio_rows = [r for r in rows if r["model"] in cfg["bio"]]
    all_rows = [r for r in rows]
    hm["clusters_FULL_bio_models"] = describe_clusters(bio_rows, std["FULL"][0](bio_rows),
                                                       std["FULL"][1], rng)
    hm["clusters_STORY_all_models"] = describe_clusters(all_rows, std["STORY"][0](all_rows),
                                                        std["STORY"][1], rng)
    hm["seed"] = args.seed
    with open(os.path.join(out, "head_matching.json"), "w") as f:
        json.dump(hm, f, indent=2)

    nm = neuron_matching(cfg, task3, args.seed)
    nm["seed"] = args.seed
    with open(os.path.join(out, "neuron_matching.json"), "w") as f:
        json.dump(nm, f, indent=2)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"head_fraction_recur": {k: v["fraction_recur"] for k, v in hm["pairs"].items()},
                   "neuron": {k: {lab: {m: v[lab][m] for m in ["frac_above_0.5", "frac_above_0.8"]}
                                  for lab in ["seed1", "null_untrained"]}
                              for k, v in nm.items() if k != "seed"},
                   "wall_seconds": time.time() - t0}, f, indent=2)
    plots(hm, nm, out)
    print("done", round(time.time() - t0), "s")


def plots(hm, nm, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(hm["pairs"])
    ax.bar(range(len(names)), [hm["pairs"][n]["fraction_recur"] for n in names], color="C2")
    ax.axhline(0.05, color="k", ls=":", lw=0.8, label="chance (5%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({hm['pairs'][n]['feature_set']})" for n in names], fontsize=7)
    ax.set_ylabel("fraction of heads that recur")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "head_recurrence.png"), dpi=110)
    plt.close(fig)

    pairs = [k for k in nm if k != "seed"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, thr in zip(axs, ["0.5", "0.8"]):
        for i, name in enumerate(pairs):
            for lab, ls in [("seed1", "-"), ("null_untrained", ":")]:
                ax.plot(range(C.N_LAYER),
                        [d[f"frac_above_{thr}"] for d in nm[name][lab]["by_layer"]],
                        ls, marker="o", color=f"C{i}",
                        label=f"{name} {'seed 1' if lab == 'seed1' else 'null'}")
        ax.set_xlabel("layer")
        ax.set_ylabel(f"fraction of neurons with best corr > {thr}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "neuron_recurrence.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
