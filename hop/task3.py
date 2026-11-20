"""Hop task 3: the swap on the explicit prompt (positive control). Gate H2.

HOP2 prompts ("The language most commonly spoken in {country} is"); P = the country token and every
later position. Swap countries: hop/swap.py swap_countries (assigned here for every eligible item
and saved to swaps.json, reused by Task 4).
1. Scale: on the 10 DEV items, s in {1, 2, 4, 8}; s* = the smallest s with flip rate >= 0.5 over
   (item, C') pairs (flip: top-1 becomes the answer token of C''s language); 8 if none.
2. Control on TEST at s*: swap, random inject (suppress + random REF-active features, matched in
   count per layer and total injected activation; one draw per pair, seed 0), suppress only.
   Flip rate and mean change in logp(answer of C') - logp(answer of C).
Usage: python hop/task3.py --seed 0
Writes results/hop/task3/{metrics.json, pairs.csv, swaps.json, cmd.txt}.
"""
import argparse
import csv
import json
import os
import random
import statistics

import common
import data
import swap
import torch

SCALES = [1, 2, 4, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task3")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    kept, _ = data.country_table(tok)
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    elig = set(sum(t1["included_ids"].values(), []))
    its = {it["id"]: it for it in data.items(tok) if it["id"] in elig}
    swaps = swap.swap_countries(list(its.values()), kept, t1["countries_hop2_correct"], args.seed)
    common.dump(swaps, os.path.join(d, "swaps.json"))
    rm = common.load_replacement_model()
    ctx = swap.Context(tok, rm)
    rng = random.Random(args.seed)
    rows = []

    def evaluate(it, C2, cond, s, ids, P, acts, base_diff):
        C = it["country"]
        sup = ctx.suppress(acts, ctx.cf[C], P)
        real = ctx.inject_feats(C, C2, s)
        if cond == "swap":
            ivs = sup + ctx.place(real, P)
        elif cond == "random_inject":
            ivs = sup + ctx.place(ctx.random_inject_feats(real, C, C2, rng), P)
        else:  # suppress
            ivs = sup
        lp = torch.log_softmax(ctx.run(ids, ivs), -1)
        a2 = kept[C2]["lang_token"]
        r = {"id": it["id"], "split": "DEV" if it["id"] in t1["DEV"] else "TEST", "country": C, "swap": C2,
             "condition": cond, "s": s, "n_suppressed": len(sup), "n_injected_features": len(real),
             "injected_total": round(sum(v for _, _, v in real), 4),
             "top1": tok.decode([int(lp.argmax())]), "flip": int(int(lp.argmax()) == a2),
             "delta_logp_diff": float(lp[a2] - lp[it["lang_token"]]) - base_diff}
        rows.append(r)
        return r

    def prep(it):
        ids = ctx.ids(it["HOP2"])
        P = data.hop2_positions(tok, it, ids[0].tolist())
        acts = ctx.acts(ids)
        return ids, P, acts

    # 1. scale on DEV
    flip_by_s = {}
    for s in SCALES:
        fl = []
        for iid in t1["DEV"]:
            it = its[iid]
            ids, P, acts = prep(it)
            lp0 = torch.log_softmax(ctx.run(ids, []), -1)
            for C2 in swaps[iid]:
                base = float(lp0[kept[C2]["lang_token"]] - lp0[it["lang_token"]])
                fl.append(evaluate(it, C2, "swap", s, ids, P, acts, base)["flip"])
        flip_by_s[s] = sum(fl) / len(fl)
        print("s", s, flip_by_s[s], flush=True)
    s_star = next((s for s in SCALES if flip_by_s[s] >= 0.5), 8)
    print("s*", s_star, flush=True)
    common.dump({"s_star": s_star, "dev_flip_rate_by_s": flip_by_s}, os.path.join(d, "s_star.json"))

    # 2. control on TEST
    for iid in t1["TEST"]:
        it = its[iid]
        ids, P, acts = prep(it)
        lp0 = torch.log_softmax(ctx.run(ids, []), -1)
        for C2 in swaps[iid]:
            base = float(lp0[kept[C2]["lang_token"]] - lp0[it["lang_token"]])
            for cond in ("swap", "random_inject", "suppress"):
                evaluate(it, C2, cond, s_star, ids, P, acts, base)
        print(iid, [r["flip"] for r in rows[-3:]], flush=True)
    common.free()

    with open(os.path.join(d, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    test = {}
    for cond in ("swap", "random_inject", "suppress"):
        rr = [r for r in rows if r["split"] == "TEST" and r["condition"] == cond]
        test[cond] = {"n_pairs": len(rr), "flip_rate": sum(r["flip"] for r in rr) / len(rr) if rr else None,
                      "mean_delta_logp_diff": statistics.mean(r["delta_logp_diff"] for r in rr) if rr else None,
                      "pairs_with_no_injected_feature": sum(r["n_injected_features"] == 0 for r in rr),
                      "pairs_with_no_suppressed_feature": sum(r["n_suppressed"] == 0 for r in rr)}
    m = {"seed": args.seed, "scales": SCALES, "dev_flip_rate_by_s": flip_by_s, "s_star": s_star,
         "n_dev_items": len(t1["DEV"]), "n_test_items": len(t1["TEST"]), "test": test,
         "gate_H2": {"test_swap_flip_ge_0.5": test["swap"]["flip_rate"] is not None and test["swap"]["flip_rate"] >= 0.5,
                     "random_inject_flip_le_0.1": test["random_inject"]["flip_rate"] is not None
                     and test["random_inject"]["flip_rate"] <= 0.1}}
    m["gate_H2"]["pass"] = all(m["gate_H2"].values())
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
