"""Hop task 5b: last-token patching (hop/PLAN.md section 7, revision 2).

For every eligible TWO item in TARGET, DESC, DESC2 and CAP TEST, 3 donors drawn with seed 0 (one
random.Random(0), recipients in the order TARGET, DESC, DESC2, CAP TEST, rng.sample over the sorted
candidate ids): eligible TWO items of the same pool (CAP TEST, or DESC + DESC2; TARGET uses the
DESC + DESC2 pool) whose language differs from the recipient's.
For each layer l in 0..25: run the recipient with its residual stream at the last position only,
after layer l (blocks.l.hook_resid_post), replaced by the donor's state at the donor's last
position (all 26 layers in one batch of 26, one patched layer per batch row). Top-1 is classed as
the donor's language, the recipient's language, or other.
Per set: donor-answer rate by layer and L_patch = first layer with rate >= 0.5 (or None).
Plain TransformerLens model, float32 on MPS.
Usage: python hop/task5b.py --seed 0
Writes results/hop/task5b/{metrics.json, pairs.csv, cmd.txt}; the plot comes from hop/plots.py.
"""
import argparse
import csv
import json
import os
import random

import common
import data
import rev2
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task5b")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    t4b = json.load(open(os.path.join(common.RESULTS, "task4b", "metrics.json")))
    its = {it["id"]: it for it in data.items(tok, desc2=True)}
    sets = {"TARGET": t1["eligible_ids"]["TARGET"], "DESC": t1["eligible_ids"]["DESC"],
            "DESC2": t4b["desc2"]["eligible_ids"], "CAP_TEST": t1["TEST"]}
    pool = {"TARGET": sets["DESC"] + sets["DESC2"], "DESC": sets["DESC"] + sets["DESC2"],
            "DESC2": sets["DESC"] + sets["DESC2"], "CAP_TEST": sets["CAP_TEST"]}
    rng = random.Random(args.seed)
    donors = {}
    for s_, ids_ in sets.items():
        for iid in ids_:
            cand = sorted(j for j in pool[s_] if its[j]["language"] != its[iid]["language"])
            donors[iid] = rng.sample(cand, 3)
    tl = rev2.load_tl()
    nL = tl.cfg.n_layers
    names = rev2.resid_names(tl)
    state = {}

    def last_states(iid):
        if iid not in state:
            ids_l = common.tokenize(tok, its[iid]["TWO"])
            _, st = rev2.run_states(tl, ids_l, {"last": len(ids_l) - 1})
            state[iid] = st["last"].to(common.DEVICE, common.DTYPE)
        return state[iid]

    rows = []
    for s_, ids_ in sets.items():
        for iid in ids_:
            rec = its[iid]
            ids = torch.tensor([common.tokenize(tok, rec["TWO"])] * nL, device=common.DEVICE)
            for dn in donors[iid]:
                dv = last_states(dn)

                def hook_for(l):
                    def fn(act, hook):
                        act[l, -1] = dv[l]
                        return act
                    return fn
                with torch.no_grad():
                    logits = tl.run_with_hooks(ids, fwd_hooks=[(names[l], hook_for(l)) for l in range(nL)])
                top = logits[:, -1].argmax(-1).cpu().tolist()
                for l in range(nL):
                    cls = ("donor" if top[l] == its[dn]["lang_token"] else
                           "recipient" if top[l] == rec["lang_token"] else "other")
                    rows.append({"set": s_, "recipient": iid, "recipient_language": rec["language"], "donor": dn,
                                 "donor_language": its[dn]["language"], "layer": l,
                                 "top1": tok.decode([top[l]]), "class": cls})
            print(iid, donors[iid], "".join({"donor": "D", "recipient": "R", "other": "."}[r["class"]]
                                            for r in rows[-3 * nL:]), flush=True)
            common.free()
    with open(os.path.join(d, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    groups = {"TARGET": ["TARGET"], "DESC": ["DESC"], "DESC2": ["DESC2"], "DESC+DESC2": ["DESC", "DESC2"],
              "CAP_TEST": ["CAP_TEST"]}
    per = {}
    for g, ss in groups.items():
        rr = [r for r in rows if r["set"] in ss]
        rate = {c: [sum(r["class"] == c for r in rr if r["layer"] == l) / sum(r["layer"] == l for r in rr)
                    for l in range(nL)] for c in ("donor", "recipient", "other")}
        per[g] = {"n_recipients": len({r["recipient"] for r in rr}), "n_pairs": len(rr) // nL,
                  "donor_rate": rate["donor"], "recipient_rate": rate["recipient"], "other_rate": rate["other"],
                  "L_patch": next((l for l in range(nL) if rate["donor"][l] >= 0.5), None)}
    m = {"seed": args.seed, "n_donors": 3, "donors": donors, "per_set": per,
         "patch_site": "blocks.l.hook_resid_post, last position only"}
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({g: {"L_patch": v["L_patch"], "n": v["n_recipients"],
                          "donor_rate": [round(x, 2) for x in v["donor_rate"]]} for g, v in per.items()}, indent=1))


if __name__ == "__main__":
    main()
