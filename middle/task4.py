"""Middle task 4: two-hop data and accuracy (middle/PLAN.md, Part 2).

Greedy first-token accuracy for every kind (HOP1, HOP2-language, HOP2-continent, TWO-language,
TWO-continent). A TWO item is eligible when the model gets its HOP1 and the matching HOP2 right.
Writes results/middle/task4/items.json (per item and kind: correct or not) used by tasks 5 and 6.

  python middle/task4.py --model dev --seed 0
"""
import argparse
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
from middle.countries import EXAMPLES, EXCLUDED, ROWS  # noqa: E402
from repeat import flow as FL  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task4")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    m, tok = FL.load(C.MODELS[args.model])
    flow = FL.Flow(m)
    its = H.items(tok)
    ok = {}
    for kind, it in its.items():
        ok[kind], _ = H.greedy_ok(flow, [x["ids"] for x in it], [x["ans"] for x in it])
    acc = {k: float(v.mean()) for k, v in ok.items()}
    elig = {}
    for name, (two, hop2) in H.TWO_KINDS.items():
        e = ok["HOP1"] & ok[hop2]
        elig[name] = {"n_items": len(e), "n_eligible": int(e.sum()),
                      "TWO_acc_all": float(ok[two].mean()),
                      "TWO_acc_eligible": float(ok[two][e].mean()) if e.any() else None,
                      "n_TWO_solved_eligible": int((ok[two] & e).sum()),
                      "n_TWO_failed_eligible": int((~ok[two] & e).sum())}
    per_item = [{"country": x["country"], "capital": x["capital"], "continent": x["continent"],
                 "language": x["language"], **{k: bool(ok[k][i]) for k in ok}}
                for i, x in enumerate(its["HOP1"])]
    with open(os.path.join(out, "items.json"), "w") as f:
        json.dump(per_item, f, indent=1)
    res = {"model": C.MODELS[args.model], "seed": args.seed, "n_rows": len(ROWS),
           "n_excluded": len(EXCLUDED), "examples": EXAMPLES, "n_items": len(per_item),
           "prompt_example": {k: tok.decode(its[k][0]["ids"]) + " -> " +
                              tok.decode([its[k][0]["ans"]]) for k in its},
           "accuracy": acc, "two_hop": elig, "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"accuracy": acc, "two_hop": elig}, indent=1))


if __name__ == "__main__":
    main()
