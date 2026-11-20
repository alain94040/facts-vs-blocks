"""Hop task 1: behaviour (hop/PLAN.md). Gate H1.

For every TARGET, DESC and CAP item, runs TWO, HOP1 and HOP2 through the ReplacementModel (whose
forward pass is the model's: Task 0 step 2) and records the top-5 next tokens with probabilities
and whether the answer token is top-1. Answer: TWO, HOP2 -> the language's answer token; HOP1 -> the
country's first token (a top-1 of " the" is therefore wrong). Eligible = all three top-1 correct.
DEV/TEST: eligible CAP items shuffled with seed 0, first 10 DEV.
Usage: python hop/task1.py --seed 0
Writes results/hop/task1/{items.csv, metrics.json, cmd.txt}.
"""
import argparse
import csv
import json
import os

import common
import data
import torch

KINDS = ["TWO", "HOP1", "HOP2"]
SETS = ["TARGET", "DESC", "CAP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    its = data.items(tok)
    rm = common.load_replacement_model()
    rows = []
    for it in its:
        r = {"id": it["id"], "set": it["set"], "country": it["country"], "language": it["language"]}
        for k in KINDS:
            ids = torch.tensor([common.tokenize(tok, it[k])], device=common.DEVICE)
            with torch.no_grad():
                p = torch.softmax(rm(ids)[0, -1].cpu().double(), -1)
            ans = it["country_first_token"] if k == "HOP1" else it["lang_token"]
            v, ix = p.topk(5)
            r[f"{k}_prompt"] = it[k]
            r[f"{k}_answer"] = tok.decode([ans])
            r[f"{k}_top1"] = tok.decode([int(ix[0])])
            r[f"{k}_correct"] = int(int(ix[0]) == ans)
            r[f"{k}_p_answer"] = round(float(p[ans]), 6)
            r[f"{k}_top5"] = json.dumps([[tok.decode([int(i)]), round(float(x), 6)] for x, i in zip(v, ix)])
        r["eligible"] = int(all(r[f"{k}_correct"] for k in KINDS))
        rows.append(r)
        print(r["id"], [r[f"{k}_top1"] for k in KINDS], r["eligible"], flush=True)
    with open(os.path.join(d, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    acc = {s: {k: sum(r[f"{k}_correct"] for r in rows if r["set"] == s) / sum(r["set"] == s for r in rows)
               for k in KINDS} for s in SETS}
    n = {s: sum(r["set"] == s for r in rows) for s in SETS}
    elig = {s: sum(r["eligible"] for r in rows if r["set"] == s) for s in SETS}
    cap_elig = [r["id"] for r in rows if r["set"] == "CAP" and r["eligible"]]
    dev, test = data.splits(cap_elig)
    # countries whose own HOP2 is top-1 correct (candidates for swap countries)
    hop2_ok = sorted({r["country"] for r in rows if r["HOP2_correct"]})
    hop2_bad = sorted({r["country"] for r in rows if not r["HOP2_correct"]})
    target = next(r for r in rows if r["id"] == "TARGET")
    m = {"seed": args.seed, "n_items": n, "accuracy": acc, "n_eligible": elig,
         "eligible_ids": {s: [r["id"] for r in rows if r["set"] == s and r["eligible"]] for s in SETS},
         "DEV": dev, "TEST": test, "countries_hop2_correct": hop2_ok, "countries_hop2_wrong": hop2_bad,
         "target": {k: target[f"{k}_top5"] for k in KINDS},
         "gate_H1": {"target_eligible": bool(target["eligible"]), "target_TWO_correct": bool(target["TWO_correct"]),
                     "n_cap_eligible": len(cap_elig), "cap_ge_25": len(cap_elig) >= 25}}
    m["gate_H1"]["pass"] = m["gate_H1"]["target_eligible"] and m["gate_H1"]["cap_ge_25"]
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({k: m[k] for k in ("accuracy", "n_eligible", "gate_H1")}, indent=1))


if __name__ == "__main__":
    main()
