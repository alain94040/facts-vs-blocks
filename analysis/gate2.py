"""Gate 2 check (PLAN.md Task 2b), written before any Task 2b result was seen.

For every run:
  - story validation loss (fixed 2,000-row sample, logged every 5% of steps) changes by less than
    0.02 nats between 90% and 100% of steps;
  - runs with biographies: overall recall on train_templates for the run's own people (its own
    set, A or B) >= 0.80, or >= 0.60 for main-32k;
  - runs with biographies: name-copy accuracy >= 0.50 (a floor, not a quality bar).
template5_heldout recall and the random-token induction ratio are reported, not gated.

Writes results/task2/gate2/{metrics.json, cmd.txt} (checks for the runs given) and rewrites
results/task2/summary.csv with one row per finished run (all runs present under results/task2/).
Exit code 1 if any checked run fails.
"""
import argparse
import csv
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["main-2k-s0", "main-8k-s0", "main-32k-s0", "ctrl-8k-x4-s0", "main-0-s0",
        "fork-A-s0", "fork-B-s0", "main-2k-s1", "main-8k-s1", "main-32k-s1", "main-0-s1"]
FLAT_MAX = 0.02
RECALL_MIN = 0.80
RECALL_MIN_32K = 0.60
NAME_COPY_MIN = 0.50


def stream_of(run):
    return run.rsplit("-s", 1)[0]


def load(run):
    path = os.path.join(ROOT, "results", "task2", run, "metrics.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        m = json.load(f)
    with open(os.path.join(ROOT, "results", "task2", "streams", stream_of(run), "metrics.json")) as f:
        meta = json.load(f)
    return m, meta


def row_for(run, m, meta):
    e = m["eval"]
    def at(frac):  # the logged point at this fraction of steps (logged at round(k * total / 20))
        pts = [r for r in m["val_curve"] if abs(r["frac"] - frac) < 1e-3]
        return pts[0]["val_sample_loss"] if pts else None
    v90, v100 = at(0.9), at(1.0)
    row = {"run": run, "stream": stream_of(run), "seed": m["seed"],
           "people_set": meta.get("people_set"), "n_people": meta["n_people"],
           "exposures": meta["exposures"], "stream_tokens": meta["stream_tokens"],
           "steps": m["total_steps"], "tokens_per_second": m["tokens_per_second"],
           "story_val_loss": e["story_val_loss"], "val90": v90, "val100": v100,
           "val_change_90_100": None if v90 is None or v100 is None else v90 - v100,
           "name_copy": e["name_copy"]["accuracy"],
           "name_copy_err_trained_last": e["name_copy"]["errors_trained_last_name"],
           "name_copy_err_other": e["name_copy"]["errors_other_token"],
           "induction_ratio": e["induction"]["ratio"]}
    for s in "AB":
        f_ = e[f"facts_set_{s}"]
        for g in ["train_templates", "template5_seen", "template5_heldout"]:
            row[f"recall_{s}_{g}"] = f_[g]["recall"]
            row[f"factloss_{s}_{g}"] = f_[g]["fact_loss"]
    own = meta.get("people_set")
    row["recall_own_train"] = row[f"recall_{own}_train_templates"] if own else None
    row["recall_own_t5_heldout"] = row[f"recall_{own}_template5_heldout"] if own else None
    return row


def checks_for(row):
    out = []
    d = row["val_change_90_100"]
    out.append({"run": row["run"], "check": "flat_90_100", "value": d, "threshold": FLAT_MAX,
                "pass": d is not None and abs(d) < FLAT_MAX})
    if row["n_people"] > 0:
        thr = RECALL_MIN_32K if row["stream"] == "main-32k" else RECALL_MIN
        out.append({"run": row["run"], "check": "recall_own_train_templates",
                    "value": row["recall_own_train"], "threshold": thr,
                    "pass": row["recall_own_train"] >= thr})
        out.append({"run": row["run"], "check": "name_copy", "value": row["name_copy"],
                    "threshold": NAME_COPY_MIN, "pass": row["name_copy"] >= NAME_COPY_MIN})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--runs", nargs="*", default=RUNS, help="runs to check")
    args = p.parse_args()

    rows = []
    for run in RUNS:
        m, meta = load(run)
        if m is not None:
            rows.append(row_for(run, m, meta))
    by_run = {r["run"]: r for r in rows}
    checks = []
    for run in args.runs:
        if run not in by_run:
            checks.append({"run": run, "check": "exists", "pass": False})
        else:
            checks += checks_for(by_run[run])

    out = os.path.join(ROOT, "results", "task2")
    os.makedirs(os.path.join(out, "gate2"), exist_ok=True)
    with open(os.path.join(out, "gate2", "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    res = {"pass": all(c["pass"] for c in checks), "runs_checked": args.runs, "checks": checks}
    with open(os.path.join(out, "gate2", "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    for run in args.runs:  # one file per run, so each run's check is kept
        rc = [c for c in checks if c["run"] == run]
        with open(os.path.join(out, "gate2", f"{run}.json"), "w") as f:
            json.dump({"pass": all(c["pass"] for c in rc), "checks": rc}, f, indent=2)
    if rows:
        with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for c in checks:
        print(("PASS " if c["pass"] else "FAIL ") + json.dumps(c))
    print("GATE 2:", "PASS" if res["pass"] else "FAIL")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
