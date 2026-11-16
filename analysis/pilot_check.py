"""Task 2a pilot pass check, written before any pilot result was seen.

Pass = overall recall on training templates (own people, set A) >= 0.80 and
name-copy accuracy >= 0.90. Writes results/task2a/pilot_check_<pilot>/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECALL_MIN = 0.80
NAME_COPY_MIN = 0.90


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", required=True, help="run name under results/task2a/, e.g. P1-main-2k-s0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    with open(os.path.join(ROOT, "results", "task2a", args.pilot, "metrics.json")) as f:
        e = json.load(f)["eval"]
    recall = e["facts_set_A"]["train_templates"]["recall"]
    copy = e["name_copy"]["accuracy"]
    res = {"pilot": args.pilot,
           "recall_A_train": recall, "recall_threshold": RECALL_MIN, "recall_pass": recall >= RECALL_MIN,
           "name_copy": copy, "name_copy_threshold": NAME_COPY_MIN, "name_copy_pass": copy >= NAME_COPY_MIN}
    res["pass"] = res["recall_pass"] and res["name_copy_pass"]
    out = os.path.join(ROOT, "results", "task2a", f"pilot_check_{args.pilot}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print("PILOT", args.pilot, "PASS" if res["pass"] else "FAIL")


if __name__ == "__main__":
    main()
