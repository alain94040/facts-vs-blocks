"""Basis task 3, step 1: the 14B outlier coordinates, found as in Repeat task 4b step 1 (16
largest mean |x| at the input of the layer at 50% of depth, first 20 TEXT SEARCH chunks,
unmodified model). Writes results/basis/task3/outliers.json.

  python basis/task3_outliers.py --seed 0
"""
import argparse
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basis import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = os.path.join(C.ROOT, "results", "basis", "task3")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd_outliers.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    W = C.weights("confirm")
    mid = round(0.5 * W.n_layers)
    top, mag = C.B4.outlier_set("confirm", mid)
    res = {"model": C.E.MODELS["confirm"], "seed": args.seed, "layer": mid,
           "outliers": top.tolist(), "outlier_mean_abs": mag[top].tolist(),
           "median_mean_abs": float(mag.median())}
    with open(os.path.join(out, "outliers.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(res)


if __name__ == "__main__":
    main()
