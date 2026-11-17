"""Basis task 0: setup checks (basis/PLAN.md).

Development model: everything of task 1 for its FAMILY zone, then the checks
  - k = d: every kept energy (own, shared, null) equals 1 to 1e-9;
  - own-basis energy >= shared-basis energy for every layer, type, setting and k (to 1e-9);
  - rotated null: own-basis energy unchanged by the rotation (to 1e-9).
7B: the same checks for one layer (FAMILY layer 10), with a two-layer zone {10, 11} for the
shared basis (a single layer's shared basis is its own basis).
Writes results/basis/zones.json and results/basis/task0/<model>/checks.json.

  python basis/task0.py --seed 0
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basis import common as C  # noqa: E402

TOL = 1e-9


def checks(curves):
    worst = {"k_eq_d_minus_1": 0.0, "own_minus_shared_min": np.inf, "rotation_own_change": 0.0}
    for key, r in curves.items():
        for name in ("own", "shared", "null"):
            if name in r:
                worst["k_eq_d_minus_1"] = max(worst["k_eq_d_minus_1"], abs(r[name][-1] - 1))
        worst["own_minus_shared_min"] = min(worst["own_minus_shared_min"],
                                            float((r["own"] - r["shared"]).min()))
        if "null_own" in r:
            worst["rotation_own_change"] = max(worst["rotation_own_change"],
                                               float(np.abs(r["null_own"] - r["own"]).max()))
    worst["pass"] = bool(worst["k_eq_d_minus_1"] < TOL and worst["own_minus_shared_min"] > -TOL
                         and worst["rotation_own_change"] < TOL)
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    t0 = time.time()
    out_root = os.path.join(C.ROOT, "results", "basis")
    os.makedirs(os.path.join(out_root, "task0"), exist_ok=True)
    with open(os.path.join(out_root, "task0", "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    zinfo = {}
    for key in ("dev", "main", "confirm"):
        W = C.weights(key)
        zinfo[C.E.MODELS[key]] = {"n_layers": W.n_layers, "d": W.D, **C.zones(W.n_layers)}
    with open(os.path.join(out_root, "zones.json"), "w") as f:
        json.dump(zinfo, f, indent=2)
    res = {"seed": args.seed}
    for key, zl in [("dev", None), ("main", {"FAMILY": [10, 11]})]:
        W = C.weights(key)
        zl = zl or {"FAMILY": C.zones(W.n_layers)["FAMILY"]}
        out = C.outliers_for(key)
        own, U, Un = C.shared_bases(W, zl, out, log=lambda m: None)
        curves = C.energies(W, zl, out, own, U, Un, cross_zone=None, log=lambda m: None)
        if key == "main":
            curves = {k: v for k, v in curves.items() if k[1] == 10}
        res[C.E.MODELS[key]] = {"zone": zl, "n_curves": len(curves), **checks(curves)}
        print(key, res[C.E.MODELS[key]], round(time.time() - t0), "s", flush=True)
    res["pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict))
    res["wall_seconds"] = time.time() - t0
    with open(os.path.join(out_root, "task0", "checks.json"), "w") as f:
        json.dump(res, f, indent=2)
    if not res["pass"]:
        with open(os.path.join(out_root, "STOP.md"), "w") as f:
            f.write("# STOP: basis task 0 checks failed\n\n```\n" + json.dumps(res, indent=2)
                    + "\n```\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
