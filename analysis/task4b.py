"""Task 4b summary: results/task4b/summary.csv and a bar chart from the per-variant metrics.

Prediction (PLAN.md): F-mlp34 or F-mlp reaches at least 80% of F-all's new-people recall (set B,
training templates), keeps set-A recall within 0.05 of the starting model, and raises STORY loss by
less than 0.02. Also reported: F-emb's new-people recall as a fraction of F-all's.

  python analysis/task4b.py --dir results/task4b
"""
import argparse
import csv
import json
import os
import shlex
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["none", "F-all", "F-attn", "F-mlp", "F-mlp34", "F-L0", "F-emb"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/task4b")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    d = os.path.join(ROOT, args.dir)
    rows = {}
    for v in VARIANTS:
        f = os.path.join(d, v, "metrics.json")
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            m = json.load(fh)
        e = m["eval"]
        rows[v] = {"variant": v, "trainable_params": m["trainable_params"],
                   "steps": m.get("steps_run", 0),
                   "new_recall_train": e["set_B_0_999"]["train_templates"]["recall"],
                   "new_recall_t5_seen": e["set_B_0_999"]["template5_seen"]["recall"],
                   "new_recall_t5_heldout": e["set_B_0_999"]["template5_heldout"]["recall"],
                   "old_recall_train": e["set_A_0_999"]["train_templates"]["recall"],
                   "old_recall_t5_heldout": e["set_A_0_999"]["template5_heldout"]["recall"],
                   "story_loss": e["story_loss"], "format_loss": e["format_loss"],
                   "name_copy": e["name_copy"], "train_loss_last10": m.get("loss_last10_mean")}
    base, ref = rows.get("none"), rows.get("F-all")
    checks = {}
    if base and ref:
        for v, r in rows.items():
            r["new_recall_frac_of_F_all"] = r["new_recall_train"] / ref["new_recall_train"]
            r["old_recall_change"] = r["old_recall_train"] - base["old_recall_train"]
            r["story_loss_change"] = r["story_loss"] - base["story_loss"]
        for v in ["F-mlp34", "F-mlp"]:
            if v in rows:
                r = rows[v]
                checks[v] = {"new_recall_ge_0.8_of_F_all": r["new_recall_frac_of_F_all"] >= 0.8,
                             "old_recall_within_0.05": abs(r["old_recall_change"]) <= 0.05,
                             "story_rise_below_0.02": r["story_loss_change"] < 0.02}
                checks[v]["meets_prediction"] = all(checks[v].values())
        if "F-emb" in rows:
            checks["F-emb"] = {"new_recall_frac_of_F_all": rows["F-emb"]["new_recall_frac_of_F_all"],
                               "at_least_0.5": rows["F-emb"]["new_recall_frac_of_F_all"] >= 0.5}
    out = list(rows.values())
    keys = []
    for r in out:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump({"variants": rows, "checks": checks, "seed": args.seed}, f, indent=2)
    with open(os.path.join(d, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    print(json.dumps(checks, indent=1))

    vs = list(rows)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(vs))
    axs[0].bar(x - 0.2, [rows[v]["new_recall_train"] for v in vs], 0.4, label="new people (set B)")
    axs[0].bar(x + 0.2, [rows[v]["old_recall_train"] for v in vs], 0.4, label="old people (set A)")
    axs[0].set_title("recall, training templates")
    axs[0].legend(fontsize=8)
    axs[1].bar(x, [rows[v]["story_loss"] for v in vs], color="C2")
    axs[1].set_ylim(min(rows[v]["story_loss"] for v in vs) - 0.05,
                    max(rows[v]["story_loss"] for v in vs) + 0.02)
    axs[1].set_title("STORY loss")
    axs[2].bar(x, [rows[v]["trainable_params"] / 1e6 for v in vs], color="C4")
    axs[2].set_title("trainable parameters (M)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(vs, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "variants.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
