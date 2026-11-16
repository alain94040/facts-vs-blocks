"""Gate 2 check (PLAN.md Task 2), written before any Task 2 results were seen.

For every run: story validation loss (fixed 2,000-row sample, logged every 5% of steps) changes by
less than 0.02 nats between 90% and 100% of steps; induction ratio < 0.5.
main-2k (every seed present) reaches >= 80% overall recall on training templates (set A).
Writes results/task2/gate2/{metrics.json, cmd.txt} and a summary table results/task2/summary.csv.
"""
import argparse
import csv
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["main-0-s0", "main-0-s1", "main-2k-s0", "main-2k-s1", "main-8k-s0", "main-8k-s1",
        "main-32k-s0", "main-32k-s1", "ctrl-8k-x4-s0", "fork-A-s0", "fork-B-s0"]
FLAT_MAX = 0.02
RECALL_MIN = 0.80
INDUCTION_MAX = 0.5


def val_at(curve, frac):
    return min(curve, key=lambda r: abs(r["frac"] - frac))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", nargs="*", default=RUNS)
    args = p.parse_args()
    rows, checks = [], []
    for run in args.runs:
        path = os.path.join(ROOT, "results", "task2", run, "metrics.json")
        if not os.path.exists(path):
            checks.append({"run": run, "check": "exists", "pass": False})
            continue
        with open(path) as f:
            m = json.load(f)
        e = m["eval"]
        v90, v100 = val_at(m["val_curve"], 0.9), val_at(m["val_curve"], 1.0)
        assert abs(v90["frac"] - 0.9) < 1e-3 and abs(v100["frac"] - 1.0) < 1e-9, run
        delta = v90["val_sample_loss"] - v100["val_sample_loss"]
        row = {"run": run, "steps": m["total_steps"], "story_val_loss": e["story_val_loss"],
               "val90": v90["val_sample_loss"], "val100": v100["val_sample_loss"],
               "val_change_90_100": delta, "induction_ratio": e["induction"]["ratio"]}
        for s in "AB":
            f_ = e[f"facts_set_{s}"]
            row[f"recall_{s}_train"] = f_["train_templates"]["recall"]
            row[f"recall_{s}_heldout"] = f_["heldout_template"]["recall"]
            row[f"factloss_{s}_train"] = f_["train_templates"]["fact_loss"]
            row[f"factloss_{s}_heldout"] = f_["heldout_template"]["fact_loss"]
        rows.append(row)
        checks.append({"run": run, "check": "flat_90_100", "value": abs(delta),
                       "threshold": FLAT_MAX, "pass": abs(delta) < FLAT_MAX})
        checks.append({"run": run, "check": "induction_ratio", "value": row["induction_ratio"],
                       "threshold": INDUCTION_MAX, "pass": row["induction_ratio"] < INDUCTION_MAX})
        if run.startswith("main-2k"):
            checks.append({"run": run, "check": "recall_A_train", "value": row["recall_A_train"],
                           "threshold": RECALL_MIN, "pass": row["recall_A_train"] >= RECALL_MIN})
    out = os.path.join(ROOT, "results", "task2")
    os.makedirs(os.path.join(out, "gate2"), exist_ok=True)
    with open(os.path.join(out, "gate2", "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    res = {"pass": all(c["pass"] for c in checks), "checks": checks}
    with open(os.path.join(out, "gate2", "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + json.dumps(c))
    print("GATE 2:", "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
