"""Hop task 5: where the bridge appears (logit lens and features by layer), hop/PLAN.md.

For TARGET and every eligible TWO prompt (TARGET, DESC, CAP), at the last position and at the
description's last token (" States" for TARGET, the capital's last token for CAP), apply Gemma's final
norm and unembedding to the residual stream after each layer, with the final logit soft-capping.
Rank of a token = number of vocabulary tokens with a strictly larger logit (0 = top), as in
Experiment E. L_bridge / L_answer = first layer after which the bridge country's first token / the
answer token has rank < 10 (or None).
Also, per item: earliest and median layer of its active M features (Task 2), and whether the
earliest fires before L_answer at the last position (a count, no test).
Usage: python hop/task5.py --seed 0
Writes results/hop/task5/{metrics.json, items.csv, ranks.json, target_logit_lens.png, cmd.txt}.
"""
import argparse
import csv
import json
import os
import statistics

import common
import data
import numpy as np
import torch


def first_top10(ranks):
    return next((l for l, r in enumerate(ranks) if r < 10), None)


def describe(v):
    x = [a for a in v if a is not None]
    return {"n": len(v), "n_none": len(v) - len(x), "median": statistics.median(x) if x else None,
            "min": min(x) if x else None, "max": max(x) if x else None,
            "counts": {str(k): v.count(k) for k in sorted(set(x))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task5")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    bridge = json.load(open(os.path.join(common.RESULTS, "task2", "bridge_sets.json")))
    use = t1["included_ids"]["TARGET"] + t1["included_ids"]["DESC"] + t1["included_ids"]["CAP"]
    its = {it["id"]: it for it in data.items(tok)}
    rm = common.load_replacement_model()
    cap = rm.cfg.output_logits_soft_cap
    names = [f"blocks.{l}.hook_resid_post" for l in range(rm.cfg.n_layers)]
    rows, ranks_all = [], {}
    for iid in use:
        it = its[iid]
        ids_l = common.tokenize(tok, it["TWO"])
        P = data.two_positions(tok, it, ids_l)
        pos = {"last": len(ids_l) - 1, "desc_last": P[0]}
        with torch.no_grad():
            _, cache = rm.run_with_cache(torch.tensor([ids_l], device=common.DEVICE),
                                         names_filter=lambda n: n in names)
            rk = {}
            for pn, p in pos.items():
                x = torch.stack([cache[n][0, p] for n in names])[:, None]  # [layers, 1, d]
                z = rm.unembed(rm.ln_final(x))[:, 0].cpu().double()
                if cap and cap > 0:
                    z = cap * torch.tanh(z / cap)
                for tn, t in (("bridge", it["country_first_token"]), ("answer", it["lang_token"])):
                    rk[f"{pn}:{tn}"] = (z > z[:, t:t + 1]).sum(1).tolist()
        del cache
        ranks_all[iid] = rk
        M = bridge[iid]["M"]
        ml = sorted(e["layer"] for e in M)
        r = {"id": iid, "set": "CAP_" + ("DEV" if iid in t1["DEV"] else "TEST") if it["set"] == "CAP" else it["set"],
             "country": it["country"]}
        for pn in pos:
            r[f"L_bridge_{pn}"] = first_top10(rk[f"{pn}:bridge"])
            r[f"L_answer_{pn}"] = first_top10(rk[f"{pn}:answer"])
        r["n_M"] = len(M)
        r["M_earliest_layer"] = ml[0] if ml else None
        r["M_median_layer"] = statistics.median(ml) if ml else None
        r["M_before_L_answer_last"] = (None if not ml or r["L_answer_last"] is None
                                       else int(ml[0] < r["L_answer_last"]))
        rows.append(r)
        print(r, flush=True)
    with open(os.path.join(d, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    common.dump(ranks_all, os.path.join(d, "ranks.json"))

    groups = {"TARGET": ["TARGET"], "DESC": ["DESC"], "CAP": ["CAP_DEV", "CAP_TEST"], "CAP_TEST": ["CAP_TEST"]}
    summ = {}
    for g, sets in groups.items():
        rr = [r for r in rows if r["set"] in sets]
        if not rr:
            continue
        summ[g] = {k: describe([r[k] for r in rr]) for k in rows[0]
                   if k.startswith("L_") or k in ("M_earliest_layer", "M_median_layer")}
        summ[g]["M_before_L_answer_last"] = {
            "yes": sum(r["M_before_L_answer_last"] == 1 for r in rr),
            "no": sum(r["M_before_L_answer_last"] == 0 for r in rr),
            "M_empty": sum(r["n_M"] == 0 for r in rr),
            "L_answer_none_with_M": sum(r["n_M"] > 0 and r["L_answer_last"] is None for r in rr)}
    m = {"seed": args.seed, "soft_cap": cap, "positions": "last and description's last token",
         "rank": "number of tokens with a strictly larger logit; top 10 = rank < 10", "per_set": summ}
    common.dump(m, os.path.join(d, "metrics.json"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rk = ranks_all["TARGET"]
    fig, ax = plt.subplots(figsize=(7, 4))
    L = np.arange(rm.cfg.n_layers)
    for key, style, lab in (("last:bridge", "o-", '" Mexico" at last token (" is")'),
                            ("last:answer", "s-", '" Spanish" at last token (" is")'),
                            ("desc_last:bridge", "o--", '" Mexico" at " States"'),
                            ("desc_last:answer", "s--", '" Spanish" at " States"')):
        ax.plot(L, np.array(rk[key]) + 1, style, ms=3, label=lab)
    ax.axhline(10, color="k", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("layer (residual stream after the layer)")
    ax.set_ylabel("rank + 1 (log scale; 1 = top)")
    ax.set_title("TARGET, logit lens: " + its["TARGET"]["TWO"], fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "target_logit_lens.png"), dpi=130)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
