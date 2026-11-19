"""Circuit task 0, step 5: effect of max_feature_nodes = 8192 on the development model's graphs.

Compares results/circuit/task0/dev_r3_cap8192 with dev_r3_nocap (same 20 prompts, same settings
except the cap; the first uncapped run saved no feature sets, so the uncapped graphs were rebuilt
with batch_size 64, which does not change a graph). Reported, not gated.

Jaccard overlap of the pruned graphs' feature sets, per prompt:
  - nodes: features identified by (layer, position, feature index), as the graph's nodes;
  - features: by (layer, feature index), any position (the plan's feature identity in Task 4.2).

Usage: python circuit/task0_cap.py --seed 0
Writes results/circuit/task0/cap_effect/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "circuit", "task0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capped", default="dev_r3_cap8192")
    ap.add_argument("--uncapped", default="dev_r3_nocap")
    args = ap.parse_args()
    runs = {}
    for k, name in (("capped", args.capped), ("uncapped", args.uncapped)):
        runs[k] = {"metrics": json.load(open(os.path.join(D, name, "metrics.json"))),
                   "features": json.load(open(os.path.join(D, name, "pruned_features.json")))}
    per = []
    for pc, pu in zip(runs["capped"]["metrics"]["per_prompt"], runs["uncapped"]["metrics"]["per_prompt"]):
        s = str(pc["seed"])
        assert pc["seed"] == pu["seed"]
        a = {tuple(x) for x in runs["capped"]["features"][s]}
        b = {tuple(x) for x in runs["uncapped"]["features"][s]}
        fa, fb = {(x[0], x[2]) for x in a}, {(x[0], x[2]) for x in b}
        per.append({"seed": pc["seed"],
                    "jaccard_nodes": len(a & b) / len(a | b), "jaccard_features": len(fa & fb) / len(fa | fb),
                    "n_pruned_nodes_capped": len(a), "n_pruned_nodes_uncapped": len(b),
                    "n_active_features": pu["n_active_features"],
                    "completeness_capped": pc["completeness_score"], "completeness_uncapped": pu["completeness_score"],
                    "replacement_capped": pc["replacement_score"], "replacement_uncapped": pu["replacement_score"],
                    "seconds_capped": pc["graph_seconds"], "seconds_uncapped": pu["graph_seconds"]})
    med = lambda k: statistics.median(p[k] for p in per)  # noqa: E731
    m = {"seed": args.seed, "capped_run": args.capped, "uncapped_run": args.uncapped,
         "max_feature_nodes": runs["capped"]["metrics"]["max_feature_nodes"], "n": len(per),
         "median": {k: med(k) for k in per[0] if k != "seed"}, "per_prompt": per}
    out = os.path.join(D, "cap_effect")
    os.makedirs(out, exist_ok=True)
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    print(json.dumps(m["median"], indent=1))


if __name__ == "__main__":
    main()
