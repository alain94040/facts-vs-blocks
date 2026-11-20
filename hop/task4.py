"""Hop task 4: the swap on the two-hop prompts (hop/PLAN.md).

Every eligible TWO prompt of TARGET, DESC and CAP TEST; P = the intervention positions
(description's last token and every later position); scale s* from Task 3; swap countries from
Task 3 (swaps.json); M(item) from Task 2 (bridge_sets.json).
Conditions:
  swap            suppress M(item) (each feature at the positions in P where it is active), inject C'
  suppress        suppress M(item) only
  random_suppress zero the same number of features active at the same (layer, position) cells, not
                  in M, matched in total activation (a uniform draw per cell is accepted when its total
                  is within 10% of M's; otherwise the closest of 2,000 draws is kept); 20 draws
  random_inject   suppress M, inject random features as in Task 3
  swap_last       swap with P = the last position only
  swap_desc       swap with P = the intervention positions except the last
Margin: the answer token's logit minus the mean logit of the answer tokens of all other languages in
the country table (final soft-capped logits). Residual = intervened margin / original margin.
Flip: top-1 becomes the answer token of C''s language. Items with an empty M are counted and left
out of the rates.
Usage: python hop/task4.py --seed 0
Writes results/hop/task4/{metrics.json, rows.csv, top5.json, cmd.txt}.
"""
import argparse
import csv
import json
import os
import random

import common
import data
import numpy as np
import swap
import torch

N_RANDOM = 20


def readings(F, R, R_rand, m_empty):
    """Decision table of hop/PLAN.md section 4 (Gate H2 passed, else Task 4 does not run)."""
    if m_empty:
        r = "M empty"
    elif F is None or R is None:
        r = "no data"
    elif F >= 0.5:
        r = "Hop" if R <= 0.5 else "Hop plus shortcut"
    else:
        r = "Shortcut" if R > 0.5 else "Inconclusive"
    void = R_rand is not None and R_rand <= 0.5
    return {"F": F, "R": R, "random_suppress_R": R_rand, "reading": r, "R_reading_void": void}


def q(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75))} if len(v) else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task4")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    kept, _ = data.country_table(tok)
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    t3 = json.load(open(os.path.join(common.RESULTS, "task3", "metrics.json")))
    bridge = json.load(open(os.path.join(common.RESULTS, "task2", "bridge_sets.json")))
    swaps = swap.load_swaps()
    s = t3["s_star"]
    use = t1["included_ids"]["TARGET"] + t1["included_ids"]["DESC"] + t1["TEST"]
    its = {it["id"]: it for it in data.items(tok)}
    langs = sorted({v["language"] for v in kept.values()})
    lang_tok = {l_: next(v["lang_token"] for v in kept.values() if v["language"] == l_) for l_ in langs}
    rm = common.load_replacement_model()
    ctx = swap.Context(tok, rm)
    rng = random.Random(args.seed)
    rows, top5, empty, hist = [], {}, {}, {}

    for iid in use:
        it = its[iid]
        st = "CAP_TEST" if it["set"] == "CAP" else it["set"]
        C = it["country"]
        M = bridge[iid]["M"]
        if not M:
            empty.setdefault(st, []).append(iid)
            continue
        for e in M:
            hist.setdefault(st, []).append(e["layer"])
        ids = ctx.ids(it["TWO"])
        P = bridge[iid]["positions"]
        last = P[-1]
        acts = ctx.acts(ids)
        others = torch.tensor([lang_tok[l_] for l_ in langs if l_ != it["language"]])

        def margin(z):
            return float(z[it["lang_token"]] - z[others].mean())

        def top(z):
            p = torch.softmax(z, -1)
            v, ix = p.topk(5)
            return [[tok.decode([int(i)]), round(float(x), 5)] for x, i in zip(v, ix)]

        z0 = ctx.run(ids, [])
        m0 = margin(z0)
        top5[iid] = {"original": top(z0)}

        def sup(pos):
            return [(e["layer"], p, e["feature"], 0.0) for e in M for p in e["positions"] if p in pos]

        def record(cond, z, C2=None, draw=None, extra=None):
            r = {"id": iid, "set": st, "country": C, "condition": cond, "swap": C2, "draw": draw,
                 "top1": tok.decode([int(z.argmax())]), "margin_original": m0, "margin": margin(z),
                 "residual": margin(z) / m0,
                 "flip": int(int(z.argmax()) == kept[C2]["lang_token"]) if C2 else None}
            r.update(extra or {})
            rows.append(r)
            return r

        # suppress only
        z = ctx.run(ids, sup(P))
        record("suppress", z, extra={"n_ivs": len(sup(P))})
        top5[iid]["suppress"] = top(z)

        # random_suppress: same (layer, position) cells, not in M, matched total activation
        Mset = {(e["layer"], e["feature"]) for e in M}
        cells = {}
        for e in M:
            for p in e["positions"]:
                cells.setdefault((e["layer"], p), []).append(e["feature"])
        tot_M = sum(float(acts[l, p, i]) for (l, p), fs in cells.items() for i in fs)
        cand = {(l, p): [int(i) for i in np.flatnonzero(acts[l, p] > 0) if (l, int(i)) not in Mset]
                for (l, p) in cells}
        for k in range(N_RANDOM):
            best = None
            for _ in range(2000):
                pick = [(l, p, i) for (l, p), fs in cells.items()
                        for i in rng.sample(cand[(l, p)], min(len(fs), len(cand[(l, p)])))]
                tot = sum(float(acts[l, p, i]) for l, p, i in pick)
                if best is None or abs(tot - tot_M) < abs(best[1] - tot_M):
                    best = (pick, tot)
                if abs(tot - tot_M) <= 0.1 * tot_M:
                    break
            pick, tot = best
            z = ctx.run(ids, [(l, p, i, 0.0) for l, p, i in pick])
            record("random_suppress", z, draw=k, extra={"n_ivs": len(pick), "total_ratio": tot / tot_M})

        for C2 in swaps[iid]:
            real = ctx.inject_feats(C, C2, s)
            ex = {"n_injected_features": len(real)}
            for cond, pos in (("swap", P), ("swap_last", [last]), ("swap_desc", P[:-1])):
                z = ctx.run(ids, sup(pos) + ctx.place(real, pos))
                record(cond, z, C2, extra=ex)
                top5[iid][f"{cond}:{C2}"] = top(z)
            z = ctx.run(ids, sup(P) + ctx.place(ctx.random_inject_feats(real, C, C2, rng), P))
            record("random_inject", z, C2, extra=ex)
            top5[iid][f"random_inject:{C2}"] = top(z)
        print(iid, len(M), top5[iid]["suppress"][0], [r["flip"] for r in rows[-4 * len(swaps[iid]):]], flush=True)
        common.free()

    with open(os.path.join(d, "rows.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    common.dump(top5, os.path.join(d, "top5.json"))

    per_set = {}
    for st in ("TARGET", "DESC", "CAP_TEST"):
        rr = [r for r in rows if r["set"] == st]
        n_items = len({r["id"] for r in rr})
        flips = {c: (lambda x: sum(x) / len(x) if x else None)([r["flip"] for r in rr if r["condition"] == c])
                 for c in ("swap", "random_inject", "swap_last", "swap_desc")}
        res_sup = [r["residual"] for r in rr if r["condition"] == "suppress"]
        res_rand = [float(np.mean([r["residual"] for r in rr if r["condition"] == "random_suppress" and r["id"] == i]))
                    for i in sorted({r["id"] for r in rr})]
        per_set[st] = {"n_items_with_M": n_items, "n_items_empty_M": len(empty.get(st, [])),
                       "items_empty_M": empty.get(st, []),
                       "n_pairs": sum(r["condition"] == "swap" for r in rr), "flip_rate": flips,
                       "residual_suppress": q(res_sup), "residual_random_suppress_item_means": q(res_rand),
                       "random_suppress_total_ratio": q([r["total_ratio"] for r in rr if r["condition"] == "random_suppress"]),
                       "M_layer_histogram": np.bincount(hist.get(st, []), minlength=rm.cfg.n_layers).tolist()}
    dec = {}
    for st in ("TARGET", "CAP_TEST"):
        p = per_set[st]
        dec[st] = readings(p["flip_rate"]["swap"], p["residual_suppress"].get("median"),
                           p["residual_random_suppress_item_means"].get("median"),
                           p["n_items_with_M"] == 0 and p["n_items_empty_M"] > 0)
    m = {"seed": args.seed, "s_star": s, "n_random_suppress_draws": N_RANDOM, "per_set": per_set, "decision": dec}
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({"decision": dec, "per_set": {k: {kk: vv for kk, vv in v.items() if kk != "M_layer_histogram"}
                                                   for k, v in per_set.items()}}, indent=1))


if __name__ == "__main__":
    main()
