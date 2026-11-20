"""Plots for Experiment H tasks 4b, 5 and 5b, drawn from their saved JSON outputs only.

circuit/.venv (used for every model run, unchanged as the plan requires) has no matplotlib, so this
script runs under the repository's main .venv:  .venv/bin/python hop/plots.py --seed 0
Writes results/hop/task4b/{probe_control.png, target_probe.png}, results/hop/task5/target_logit_lens.png,
results/hop/task5b/patching.png.
"""
import argparse
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "hop")


def load(*p):
    return json.load(open(os.path.join(R, *p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()

    # Task 4b: control by layer
    m = load("task4b", "metrics.json")
    c = m["control_by_layer"]
    L = np.array([x["layer"] for x in c])
    ls = m["gate_H3"]["l_star"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(L, [x["top1_acc"] for x in c], "o-", ms=3, label="top-1 accuracy, all countries")
    ax.plot(L, [x["within_lang_acc"] for x in c], "s-", ms=3, label="within-language accuracy")
    ax.plot(L, [x["chance_top1"] for x in c], ":", color="C0", label="chance, top-1")
    ax.plot(L, [x["chance_within"] for x in c], ":", color="C1", label="chance, within-language")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.axvline(ls, color="k", lw=0.5, ls="--")
    ax.set_xlabel("layer (residual stream after the layer), last position")
    ax.set_ylabel("leave-one-template-out accuracy")
    ax.set_title(f"Country probe control (gate H3); l* = {ls}", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "task4b", "probe_control.png"), dpi=130)
    plt.close(fig)

    # Task 4b: TARGET panel
    t, bl = m["target"]["last"], m["by_layer"]["DESC+DESC2"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(L, t["mexico_rank_all_by_layer"], "o-", ms=3, label="TARGET: Mexico rank, all countries")
    ax.plot(L, t["mexico_rank_within_spanish_by_layer"], "s-", ms=3,
            label=f"TARGET: Mexico rank within Spanish (k={t['k_spanish']})")
    ax.plot(L, bl["last"]["median_rank_within"], "^--", ms=3,
            label=f"DESC + DESC2: median within-language rank (n={bl['n']})")
    ax.axvline(ls, color="k", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("layer (residual stream after the layer), last position")
    ax.set_ylabel("rank (log scale; 1 = top)")
    ax.set_title(f"Class-mean country probe on TWO prompts (l* = {ls} dotted)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "task4b", "target_probe.png"), dpi=130)
    plt.close(fig)

    # Task 5: TARGET logit lens
    rk = load("task5", "ranks.json")["TARGET"]
    L5 = np.arange(len(rk["last:bridge"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, style, lab in (("last:bridge", "o-", '" Mexico" at last token (" is")'),
                            ("last:answer", "s-", '" Spanish" at last token (" is")'),
                            ("desc_last:bridge", "o--", '" Mexico" at " States"'),
                            ("desc_last:answer", "s--", '" Spanish" at " States"')):
        ax.plot(L5, np.array(rk[key]) + 1, style, ms=3, label=lab)
    ax.axhline(10, color="k", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("layer (residual stream after the layer)")
    ax.set_ylabel("rank + 1 (log scale; 1 = top)")
    ax.set_title("TARGET, logit lens: The language most commonly spoken in the country south of the "
                 "United States is", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "task5", "target_logit_lens.png"), dpi=130)
    plt.close(fig)

    # Task 5b: patching, with Task 5's median L_answer (last position)
    p = load("task5b", "metrics.json")["per_set"]
    t5 = load("task5", "metrics.json")["per_set"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, (g, g5) in enumerate((("TARGET", "TARGET"), ("DESC", "DESC"), ("DESC2", "DESC2"),
                                 ("CAP_TEST", "CAP_TEST"))):
        if g not in p:
            continue
        ax.plot(np.arange(len(p[g]["donor_rate"])), p[g]["donor_rate"], "o-", ms=3, color=f"C{k}",
                label=f"{g} (n={p[g]['n_recipients']}), L_patch={p[g]['L_patch']}")
        la = t5.get(g5, {}).get("L_answer_last", {}).get("median")
        if la is not None:
            ax.axvline(la, color=f"C{k}", lw=0.8, ls=":")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.set_xlabel("patched layer (residual stream after the layer, last position)")
    ax.set_ylabel("donor-answer rate")
    ax.set_title("Last-token patching; dotted: Task 5 median L_answer (last position)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(R, "task5b", "patching.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
