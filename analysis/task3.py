"""Task 3: which parts hold the facts (PLAN.md Phase 2).

Per run: importance rankings (I_FACT, I_FORMAT, I_STORY) for 6,144 neurons and 48 heads; neuron
ablation curves for the top k by I_FACT, by I_STORY and for k random neurons (3 draws), k = 8..4096;
knowledge set K; overlaps; single-head ablations; effective rank of every weight matrix.

  python analysis/task3.py --runs main-2k-s0 ... [--out results/task3] --seed 0
  python analysis/task3.py --summary --out results/task3     (summary.csv and cross-run plots)

Writes <out>/<run>/{metrics.json, importance.npz, cmd.txt, ablation_curves.png,
knowledge_by_layer.png, effective_rank.png}.
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

KS = [8 * 2 ** i for i in range(10)]  # 8 .. 4096
N_RANDOM = 3
N_NEURONS = C.N_LAYER * C.D_MLP
MEASURES = ["recall", "format_loss", "name_copy", "story_loss"]


def effective_rank(w):
    s = np.linalg.svd(w.detach().cpu().double().numpy(), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_ranks(model):
    out = {"tok_emb": effective_rank(model.tok_emb.weight),
           "pos_emb": effective_rank(model.pos_emb.weight)}
    for l, b in enumerate(model.blocks):
        for name, w in [("qkv", b.attn.qkv.weight), ("out", b.attn.out.weight),
                        ("up", b.mlp.up.weight), ("down", b.mlp.down.weight)]:
            out[f"L{l}.{name}"] = effective_rank(w)
    return out


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b)


def random_jaccard(k, n=N_NEURONS):
    """Expected Jaccard of two independent uniform random k-subsets of n (exact expectation)."""
    from scipy.stats import hypergeom
    i = np.arange(0, k + 1)
    pm = hypergeom(n, k, k).pmf(i)
    return float((pm * i / (2 * k - i)).sum())


def analyse(run, out_root, seed):
    t0 = time.time()
    model, meta = C.load_model(run), C.run_meta(run)
    means, n_ref = C.component_means(model)
    imp = {}
    for D in ["FACT", "FORMAT", "STORY"]:
        imp[D] = C.importance(model, means, D, meta)
    out = os.path.join(out_root, run)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, "importance.npz"),
                        **{f"neurons_{D}": imp[D][0] for D in imp},
                        **{f"heads_{D}": imp[D][1] for D in imp})
    rank = {D: np.argsort(-imp[D][0].ravel(), kind="stable") for D in imp}
    rng = np.random.default_rng(seed)
    rand_orders = [rng.permutation(N_NEURONS) for _ in range(N_RANDOM)]

    base = C.evaluate_all(model, meta)
    print(run, "base", base, flush=True)

    def curve(order):
        rows = []
        for k in KS:
            with C.Hooks(model, means, ablate_neurons=C.neuron_sets(order[:k])):
                r = C.evaluate_all(model, meta)
            rows.append({"k": k, **r})
        return rows

    curves = {"top_FACT": curve(rank["FACT"]), "top_STORY": curve(rank["STORY"])}
    for d, o in enumerate(rand_orders):
        curves[f"random_{d}"] = curve(o)
    print(run, "curves done", round(time.time() - t0), "s", flush=True)

    # knowledge set
    K = next((r["k"] for r in curves["top_FACT"] if r["recall"] <= 0.5 * base["recall"]), None)
    ks = {"K": K}
    if K is not None:
        ki = KS.index(K)
        kset = rank["FACT"][:K]
        at = curves["top_FACT"][ki]
        rnd = [curves[f"random_{d}"][ki] for d in range(N_RANDOM)]
        ks.update({
            "neurons_per_layer": np.bincount(kset // C.D_MLP, minlength=C.N_LAYER).tolist(),
            "at_K": at,
            "story_loss_change": at["story_loss"] - base["story_loss"],
            "format_loss_change": at["format_loss"] - base["format_loss"],
            "random_K_story_loss_change": [r["story_loss"] - base["story_loss"] for r in rnd],
            "random_K_format_loss_change": [r["format_loss"] - base["format_loss"] for r in rnd],
            "random_K_recall": [r["recall"] for r in rnd],
            "jaccard_vs_top_STORY": jaccard(kset, rank["STORY"][:K]),
            "jaccard_vs_top_FORMAT": jaccard(kset, rank["FORMAT"][:K]),
            "jaccard_random_expected": random_jaccard(K),
        })
        ks["random_K_story_loss_change_mean"] = float(np.mean(ks["random_K_story_loss_change"]))
        ks["random_K_format_loss_change_mean"] = float(np.mean(ks["random_K_format_loss_change"]))

    # single-head ablations
    heads = []
    for l in range(C.N_LAYER):
        for h in range(C.N_HEAD):
            with C.Hooks(model, means, ablate_heads={l: [h]}):
                r = C.evaluate_all(model, meta)
            heads.append({"layer": l, "head": h,
                          **{f"d_{m}": r[m] - base[m] for m in MEASURES},
                          "I_FACT": float(imp["FACT"][1][l, h]),
                          "I_FORMAT": float(imp["FORMAT"][1][l, h]),
                          "I_STORY": float(imp["STORY"][1][l, h])})
    er = effective_ranks(model)

    res = {"run": run, "seed": seed, "stream": C.stream_of(run), "ckpt": C.ckpt_path(run),
           "people_set": meta.get("people_set"), "n_people": meta.get("n_people", 0),
           "fact_set": C.fact_set(meta), "n_fact_prompts": len(C.fact_prompts(meta)),
           "n_reference_tokens": n_ref, "k_grid": KS, "base": base, "curves": curves,
           "knowledge_set": ks, "heads": heads, "effective_rank": er,
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    plots(res, out)
    print(run, "done", round(time.time() - t0), "s; K =", K, flush=True)
    return res


def plots(res, out):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    base = res["base"]
    for ax, m in zip(axs, MEASURES):
        for name, style in [("top_FACT", "C3"), ("top_STORY", "C0")]:
            ax.plot(KS, [r[m] for r in res["curves"][name]], "o-", color=style, label=name, ms=3)
        rnd = np.array([[r[m] for r in res["curves"][f"random_{d}"]] for d in range(N_RANDOM)])
        ax.plot(KS, rnd.mean(0), "o-", color="0.4", label="random (mean of 3)", ms=3)
        ax.fill_between(KS, rnd.min(0), rnd.max(0), color="0.8")
        ax.axhline(base[m], color="k", lw=0.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("neurons ablated (k)")
        ax.set_title(m)
        if res["knowledge_set"]["K"]:
            ax.axvline(res["knowledge_set"]["K"], color="C3", lw=0.8, ls="--")
    axs[0].legend(fontsize=8)
    fig.suptitle(f"{res['run']}: mean-ablation of neurons")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ablation_curves.png"), dpi=110)
    plt.close(fig)

    ks = res["knowledge_set"]
    if ks["K"]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(range(C.N_LAYER), ks["neurons_per_layer"], color="C3")
        ax.set_xlabel("layer")
        ax.set_ylabel("neurons in knowledge set")
        ax.set_title(f"{res['run']}: knowledge set (K = {ks['K']}) by layer")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "knowledge_by_layer.png"), dpi=110)
        plt.close(fig)

    er = res["effective_rank"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, name in enumerate(["qkv", "out", "up", "down"]):
        ax.plot(range(C.N_LAYER), [er[f"L{l}.{name}"] for l in range(C.N_LAYER)], "o-", label=name)
    ax.set_xlabel("layer")
    ax.set_ylabel("effective rank")
    ax.set_title(f"{res['run']}: effective rank (tok_emb {er['tok_emb']:.0f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "effective_rank.png"), dpi=110)
    plt.close(fig)


def summary(out_root):
    rows, res = [], {}
    for run in sorted(os.listdir(out_root)):
        p = os.path.join(out_root, run, "metrics.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)
        res[run] = r
        ks = r["knowledge_set"]
        row = {"run": run, "n_people": r["n_people"], "people_set": r["people_set"],
               **{f"base_{m}": r["base"][m] for m in MEASURES}, "K": ks["K"]}
        for key in ["story_loss_change", "random_K_story_loss_change_mean", "format_loss_change",
                    "random_K_format_loss_change_mean", "jaccard_vs_top_STORY",
                    "jaccard_vs_top_FORMAT", "jaccard_random_expected"]:
            row[key] = ks.get(key)
        row["recall_at_K"] = ks["at_K"]["recall"] if ks["K"] else None
        row["name_copy_at_K"] = ks["at_K"]["name_copy"] if ks["K"] else None
        for l in range(C.N_LAYER):
            row[f"K_layer{l}"] = ks["neurons_per_layer"][l] if ks["K"] else None
        hd = r["heads"]
        worst = min(hd, key=lambda h: h["d_recall"])
        row["max_single_head_recall_drop"] = -worst["d_recall"]
        row["max_single_head_recall_drop_head"] = f"L{worst['layer']}H{worst['head']}"
        rows.append(row)
    if not rows:
        return
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for row in rows:
        if row["K"]:
            ax.plot(row["n_people"], row["K"], "o")
            ax.annotate(row["run"], (row["n_people"], row["K"]), fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("people in training")
    ax.set_ylabel("K (neurons to halve recall)")
    ax.set_title("Knowledge-set size against number of people")
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "K_vs_people.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.8 / len(rows)
    for i, row in enumerate(rows):
        if row["K"]:
            v = np.array([row[f"K_layer{l}"] for l in range(C.N_LAYER)]) / row["K"]
            ax.bar(np.arange(C.N_LAYER) + i * w, v, w, label=row["run"])
    ax.set_xlabel("layer")
    ax.set_ylabel("share of knowledge set")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "knowledge_by_layer.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for run, r in res.items():
        er = r["effective_rank"]
        ax.plot(range(C.N_LAYER), [er[f"L{l}.up"] for l in range(C.N_LAYER)], "o-", label=run)
    ax.set_xlabel("layer")
    ax.set_ylabel("effective rank of mlp.up")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "effective_rank_up.png"), dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--out", default="results/task3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summary", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out_root = os.path.join(C.ROOT, args.out)
    for run in args.runs:
        analyse(run, out_root, args.seed)
    if args.summary or args.runs:
        summary(out_root)
        with open(os.path.join(out_root, "cmd.txt"), "a") as f:
            f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv)
                    + "\n")


if __name__ == "__main__":
    main()
