"""Middle task 8: reuse null with the same answer (middle/PLAN.md, follow-up).

Recomputes task 5's states, L_answer1 / L_answer2 and three-layer update sums for every
eligible item, then:
  real              cos(HOP2 sum of item k, TWO sum of item k)                 (as task 5)
  null, different   cos(HOP2 sum of task 5's partner, TWO sum of k)            (task 5 pairing)
  null, same        cos(HOP2 sum of a same-answer partner, TWO sum of k)
Same-answer pairing: items in a seeded random cycle (seed 0); each item takes the next item after
it with the same answer and a HOP2 sum, skipping partners already used 3 times. Items with no
such partner (unique answer) are excluded and counted. All three cosines are reported on the
items that have a same-answer partner. Outlier coordinates removed, as in task 5.

  python middle/task8.py --model dev --seed 0
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
from middle import common as C  # noqa: E402
from middle import twohop as H  # noqa: E402
from middle.task5 import block_sum, boot_mean, cos, first_top, ranks  # noqa: E402
from repeat import flow as FL  # noqa: E402

MAX_USES = 3


def cycle_partner(perm, k, ok_fn, uses=None):
    pos = perm.index(k)
    for s in range(1, len(perm)):
        q = perm[(pos + s) % len(perm)]
        if ok_fn(q) and (uses is None or uses.get(q, 0) < MAX_USES):
            return int(q)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task8")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(C.out_dir(args.model, "task4"), "items.json")) as f:
        ok = json.load(f)
    with open(os.path.join(C.out_dir(args.model, "task5"), "items.csv")) as f:
        t5 = {(r["kind"], r["country"]): r for r in csv.DictReader(f)}
    m, tok = FL.load(C.MODELS[args.model])
    flow = FL.Flow(m)
    d = flow.cfg.hidden_size
    keep = np.setdiff1d(np.arange(d), C.outliers(args.model))
    its = H.items(tok)
    res, rows, max_dev = {}, [], 0.0
    for name, (two, hop2) in H.TWO_KINDS.items():
        idx = [i for i, o in enumerate(ok) if o["HOP1"] and o[hop2]]
        T = [its[two][i] for i in idx]
        P = [its[hop2][i] for i in idx]
        sT = H.states(flow, [x["ids"] for x in T])
        sP = H.states(flow, [x["ids"] for x in P])
        L2 = [first_top(r) for r in ranks(flow, sT, [x["ans"] for x in T])]
        L1 = [first_top(r) for r in ranks(flow, sP, [x["ans"] for x in P])]
        sums_T = [block_sum(sT[k], L2[k]) for k in range(len(idx))]
        sums_P = [block_sum(sP[k], L1[k]) for k in range(len(idx))]
        # task 5's different-answer pairing, reproduced exactly (same seed, same order of draws)
        perm = list(np.random.default_rng(args.seed).permutation(len(idx)))
        diff = [cycle_partner(perm, k, lambda q, k=k: T[q]["answer"] != T[k]["answer"]
                              and sums_P[q] is not None) for k in range(len(idx))]
        uses, same = {}, []
        for k in range(len(idx)):
            q = cycle_partner(perm, k, lambda q, k=k: q != k and T[q]["answer"] == T[k]["answer"]
                              and sums_P[q] is not None, uses)
            if q is not None:
                uses[q] = uses.get(q, 0) + 1
            same.append(q)
        real_c, diff_c, same_c = [], [], []
        n_unique = 0
        for k in range(len(idx)):
            if sums_T[k] is None or sums_P[k] is None:
                continue
            rc = cos(sums_P[k], sums_T[k], keep)
            dc = cos(sums_P[diff[k]], sums_T[k], keep) if diff[k] is not None else None
            r5 = t5[(name, T[k]["country"])]
            max_dev = max(max_dev, abs(rc - float(r5["reuse_cos"])),
                          abs(dc - float(r5["null_cos"])) if dc is not None else 0.0)
            if same[k] is None:
                n_unique += 1
                continue
            sc = cos(sums_P[same[k]], sums_T[k], keep)
            real_c.append(rc)
            diff_c.append(dc)
            same_c.append(sc)
            rows.append({"kind": name, "country": T[k]["country"], "answer": T[k]["answer"],
                         "same_partner": T[same[k]]["country"],
                         "diff_partner": T[diff[k]]["country"], "real_cos": rc,
                         "null_same_cos": sc, "null_diff_cos": dc})
        real_c, diff_c, same_c = map(np.array, (real_c, diff_c, same_c))
        res[name] = {"n_eligible": len(idx), "n_excluded_no_same_answer_partner": n_unique,
                     "n_used": int(len(real_c)), "max_partner_uses": max(uses.values()),
                     "real": boot_mean(real_c), "null_same_answer": boot_mean(same_c, 1),
                     "null_different_answer": boot_mean(diff_c, 2),
                     "real_minus_same_null": boot_mean(real_c - same_c, 3),
                     "real_minus_different_null": boot_mean(real_c - diff_c, 4)}
        print(name, json.dumps({k: res[name][k] for k in
                                ("n_used", "n_excluded_no_same_answer_partner",
                                 "real_minus_same_null")}), flush=True)
    with open(os.path.join(out, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_res = {"model": C.MODELS[args.model], "seed": args.seed, "max_uses": MAX_USES,
               "check_max_abs_diff_vs_task5": max_dev, "by_kind": res,
               "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(out_res, f, indent=1)
    print("max |difference| from task 5's cosines:", max_dev)


if __name__ == "__main__":
    main()
