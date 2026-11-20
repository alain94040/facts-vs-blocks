"""Hop task 4b: country probe (hop/PLAN.md section 7, revision 2). Gate H3.

0. DESC2 behaviour: Task 1's three checks (TWO, HOP1, HOP2; revision 1 country-rank rule) on every
   DESC2 item; eligible = TWO and HOP2 top-1 correct and HOP1 country rank 1.
1. PROBE-TRAIN prompts (P1, P2, P3 per country in the table): residual stream at the last position
   after every layer (float64, CPU). Kept when the prompt passes the country-rank rule.
2. Class-mean probe per layer: subtract the mean over the training prompts; class vector = mean of
   a country's centred states; score of country C for h = cos(h - mean, class vector of C).
3. Leave one template out (train on two templates, classify the third, all three choices; held-out
   predictions pooled over the three folds). Per layer: top-1 accuracy over all countries and
   within-language accuracy (languages with >= 2 countries in the table; true country highest
   among the countries of its language that have a class vector). A held-out prompt whose country
   has no class vector in that fold counts as wrong. Chance: 1 / n_classes and mean 1/k.
   l* = layer with the highest pooled within-language accuracy (earliest on a tie). Gate H3: at l*,
   top-1 >= 0.5 and within-language >= 0.5.
4. Test: final class vectors from all kept prompts. For every eligible TWO item in TARGET, DESC,
   DESC2 (and CAP TEST, reported separately), at the last position and at the description's last
   token, per layer: bridge rank among all countries, within-language rank, top-3 countries.
5. At l*, over eligible TARGET + DESC + DESC2 items whose language has >= 2 countries: W = number
   with within-language rank 1; p = P(W_sim >= W) over 100,000 simulated draws (item i draws rank 1
   with probability 1/k_i, seed 0); language hit rate = fraction whose top-1 country speaks the
   bridge's language. The same for CAP TEST, reported separately.
Usage: python hop/task4b.py --seed 0
Writes results/hop/task4b/{metrics.json, desc2_items.csv, probe_prompts.csv, control_by_layer.csv,
test_items.csv, test_ranks.json, cmd.txt}; the plot target_probe.png comes from hop/plots.py.
"""
import argparse
import csv
import json
import os
import statistics

import common
import data
import numpy as np
import rev2
import torch

KINDS = ["TWO", "HOP1", "HOP2"]


def class_vectors(X, y, classes):
    """X [n, d] states at one layer, y country per row. Returns (mean [d], V [n_classes, d] unit)."""
    mu = X.mean(0)
    V = np.stack([(X[[k for k, c in enumerate(y) if c == cl]] - mu).mean(0) for cl in classes])
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return mu, V


def scores(H, mu, V):
    Hc = H - mu
    Hc = Hc / np.linalg.norm(Hc, axis=-1, keepdims=True)
    return Hc @ V.T


def ranks(sc, classes, country, lang_of):
    """sc [n_classes] scores. Returns (rank among all, within-language rank, k, top-3)."""
    if country not in classes:
        return None, None, None, [classes[j] for j in np.argsort(-sc)[:3]]
    t = sc[classes.index(country)]
    same = [j for j, c in enumerate(classes) if lang_of[c] == lang_of[country]]
    return (int((sc > t).sum()) + 1, int((sc[same] > t).sum()) + 1, len(same),
            [classes[j] for j in np.argsort(-sc)[:3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task4b")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    kept, _ = data.country_table(tok)
    lang_of = {c: v["language"] for c, v in kept.items()}
    n_lang = {l_: sum(v == l_ for v in lang_of.values()) for l_ in set(lang_of.values())}
    ctoks = sorted({v["country_first_token"] for v in kept.values()})
    the = tok.encode(" the", add_special_tokens=False)[0]
    assert the not in ctoks
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    all_items = {it["id"]: it for it in data.items(tok, desc2=True)}
    tl = rev2.load_tl()
    n_layers = tl.cfg.n_layers

    # 0. DESC2 behaviour
    d2rows = []
    for it in [x for x in all_items.values() if x["set"] == "DESC2"]:
        r = {"id": it["id"], "set": "DESC2", "desc": it["desc"], "country": it["country"], "language": it["language"]}
        for k in KINDS:
            ids_l = common.tokenize(tok, it[k])
            p, _ = rev2.run_states(tl, ids_l, {})
            ans = it["country_first_token"] if k == "HOP1" else it["lang_token"]
            v, ix = p.topk(5)
            r[f"{k}_prompt"] = it[k]
            r[f"{k}_top1"] = tok.decode([int(ix[0])])
            r[f"{k}_correct"] = int(int(ix[0]) == ans)
            r[f"{k}_p_answer"] = round(float(p[ans]), 6)
            r[f"{k}_top5"] = json.dumps([[tok.decode([int(i)]), round(float(x), 6)] for x, i in zip(v, ix)])
            if k == "HOP1":
                r["HOP1_bridge_country_rank"] = rev2.country_rank(p, ans, ctoks)
                r["HOP1_country_rank_ok"] = int(r["HOP1_bridge_country_rank"] == 1)
        r["eligible"] = int(r["TWO_correct"] and r["HOP2_correct"] and r["HOP1_country_rank_ok"])
        d2rows.append(r)
        print(r["id"], [r[f"{k}_top1"] for k in KINDS], r["eligible"], flush=True)
    with open(os.path.join(d, "desc2_items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d2rows[0]))
        w.writeheader()
        w.writerows(d2rows)
    desc2_elig = [r["id"] for r in d2rows if r["eligible"]]
    desc2_m = {"n_items": len(d2rows),
               "dropped_not_in_table": [c for _, c in data.DESC2 if c not in kept],
               "accuracy": {k: sum(r[f"{k}_correct"] for r in d2rows) / len(d2rows) for k in KINDS},
               "HOP1_country_rank_rule": sum(r["HOP1_country_rank_ok"] for r in d2rows) / len(d2rows),
               "n_eligible": len(desc2_elig), "eligible_ids": desc2_elig}

    # 1. probe prompts
    pp = data.probe_prompts(tok)
    X = np.zeros((len(pp), n_layers, tl.cfg.d_model))
    for k, q in enumerate(pp):
        ids_l = common.tokenize(tok, q["text"])
        p, st = rev2.run_states(tl, ids_l, {"last": len(ids_l) - 1})
        X[k] = st["last"].numpy()
        q["country_rank"] = rev2.country_rank(p, q["country_first_token"], ctoks)
        q["top_country_token"] = tok.decode([ctoks[int(p[ctoks].argmax())]])
        q["top1"] = tok.decode([int(p.argmax())])
        q["kept"] = int(q["country_rank"] == 1)
    common.free()
    with open(os.path.join(d, "probe_prompts.csv"), "w", newline="") as f:
        keys = ["country", "language", "template", "text", "top1", "top_country_token", "country_rank", "kept"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(pp)
    keep = [k for k, q in enumerate(pp) if q["kept"]]
    per_t = {t: {"n": sum(q["template"] == t for q in pp), "kept": sum(q["template"] == t and q["kept"] for q in pp)}
             for t in data.PROBE_TEMPLATES}
    print("kept per template", per_t, flush=True)

    # 3. leave one template out
    ctrl = []
    for l in range(n_layers):
        n_top = n_within = n_all = n_wl = 0
        chance_w, n_cls = [], []
        miss = 0
        for held in data.PROBE_TEMPLATES:
            tr = [k for k in keep if pp[k]["template"] != held]
            te = [k for k in keep if pp[k]["template"] == held]
            classes = sorted({pp[k]["country"] for k in tr})
            n_cls.append(len(classes))
            mu, V = class_vectors(X[tr, l], [pp[k]["country"] for k in tr], classes)
            S = scores(X[te, l], mu, V)
            for row, k in zip(S, te):
                c = pp[k]["country"]
                r_all, r_w, kk, _ = ranks(row, classes, c, lang_of)
                n_all += 1
                if r_all is None:
                    miss += 1
                n_top += int(r_all == 1)
                if n_lang[lang_of[c]] >= 2:
                    n_wl += 1
                    n_within += int(r_w == 1)
                    chance_w.append(1 / kk if kk else 1 / n_lang[lang_of[c]])
        ctrl.append({"layer": l, "n_heldout": n_all, "top1_acc": n_top / n_all, "n_within": n_wl,
                     "within_lang_acc": n_within / n_wl, "chance_top1": float(np.mean([1 / n for n in n_cls])),
                     "chance_within": float(np.mean(chance_w)), "heldout_without_class_vector": miss})
    with open(os.path.join(d, "control_by_layer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ctrl[0]))
        w.writeheader()
        w.writerows(ctrl)
    best = max(c["within_lang_acc"] for c in ctrl)
    l_star = next(c["layer"] for c in ctrl if c["within_lang_acc"] == best)
    gate = {"l_star": l_star, "top1_acc": ctrl[l_star]["top1_acc"], "within_lang_acc": ctrl[l_star]["within_lang_acc"]}
    gate["pass"] = gate["top1_acc"] >= 0.5 and gate["within_lang_acc"] >= 0.5
    print("gate H3", gate, flush=True)

    m = {"seed": args.seed, "desc2": desc2_m, "probe_prompts_per_template": per_t,
         "probe_n_kept": len(keep), "probe_countries_with_class_vector": len({pp[k]["country"] for k in keep}),
         "countries_without_class_vector": sorted(set(kept) - {pp[k]["country"] for k in keep}),
         "extra_capitals": data.EXTRA_CAPITALS, "control_by_layer": ctrl, "gate_H3": gate}
    if not gate["pass"]:
        common.dump(m, os.path.join(d, "metrics.json"))
        print("GATE H3 FAILS")
        return

    # 4. test
    classes = sorted({pp[k]["country"] for k in keep})
    yk = [pp[k]["country"] for k in keep]
    final = [class_vectors(X[keep, l], yk, classes) for l in range(n_layers)]
    sets = {"TARGET": t1["eligible_ids"]["TARGET"], "DESC": t1["eligible_ids"]["DESC"], "DESC2": desc2_elig,
            "CAP_TEST": t1["TEST"]}
    test_rows, test_ranks = [], {}
    for st_name, ids_ in sets.items():
        for iid in ids_:
            it = all_items[iid]
            ids_l = common.tokenize(tok, it["TWO"])
            P = data.two_positions(tok, it, ids_l)
            _, st = rev2.run_states(tl, ids_l, {"last": len(ids_l) - 1, "desc_last": P[0]})
            rk = {}
            for pn, H in st.items():
                H = H.numpy()
                per = [ranks(scores(H[l][None], *final[l])[0], classes, it["country"], lang_of) for l in range(n_layers)]
                rk[pn] = {"rank_all": [x[0] for x in per], "rank_within": [x[1] for x in per], "k": per[0][2],
                          "top3": [x[3] for x in per]}
            test_ranks[iid] = rk
            r = {"id": iid, "set": st_name, "country": it["country"], "language": it["language"],
                 "k_lang": rk["last"]["k"], "desc_last_token": tok.decode([ids_l[P[0]]])}
            for pn in ("last", "desc_last"):
                r[f"{pn}_rank_all_lstar"] = rk[pn]["rank_all"][l_star]
                r[f"{pn}_rank_within_lstar"] = rk[pn]["rank_within"][l_star]
                r[f"{pn}_top3_lstar"] = json.dumps(rk[pn]["top3"][l_star])
                r[f"{pn}_top1_lang_hit_lstar"] = int(lang_of[rk[pn]["top3"][l_star][0]] == it["language"])
            test_rows.append(r)
            print(iid, r["last_rank_all_lstar"], r["last_rank_within_lstar"], r["k_lang"], r["last_top3_lstar"], flush=True)
    common.free()
    with open(os.path.join(d, "test_items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(test_rows[0]))
        w.writeheader()
        w.writerows(test_rows)
    common.dump(test_ranks, os.path.join(d, "test_ranks.json"))

    # 5. statistic
    rng = np.random.default_rng(args.seed)

    def stat(rows, pn):
        rr = [r for r in rows if r["k_lang"] is not None and r["k_lang"] >= 2]
        if not rr:
            return {"n": 0}
        W = sum(r[f"{pn}_rank_within_lstar"] == 1 for r in rr)
        pk = np.array([1 / r["k_lang"] for r in rr])
        sim = (rng.random((100_000, len(rr))) < pk).sum(1)
        return {"n": len(rr), "W": int(W), "expected_W": float(pk.sum()), "p_value": float((sim >= W).mean()),
                "language_hit_rate": sum(r[f"{pn}_top1_lang_hit_lstar"] for r in rr) / len(rr),
                "rank_all_median": statistics.median(r[f"{pn}_rank_all_lstar"] for r in rr),
                "n_rank_all_1": sum(r[f"{pn}_rank_all_lstar"] == 1 for r in rr),
                "items_excluded_k1": [r["id"] for r in rows if r not in rr]}

    main_rows = [r for r in test_rows if r["set"] in ("TARGET", "DESC", "DESC2")]
    cap_rows = [r for r in test_rows if r["set"] == "CAP_TEST"]
    res = {"main_last": stat(main_rows, "last"), "main_desc_last": stat(main_rows, "desc_last"),
           "cap_test_last": stat(cap_rows, "last"), "cap_test_desc_last": stat(cap_rows, "desc_last")}
    for s_ in ("TARGET", "DESC", "DESC2"):
        res[f"{s_}_last"] = stat([r for r in test_rows if r["set"] == s_], "last")
    tr_ = test_ranks["TARGET"]
    target = {pn: {"mexico_rank_all_by_layer": tr_[pn]["rank_all"], "mexico_rank_within_spanish_by_layer": tr_[pn]["rank_within"],
                   "k_spanish": tr_[pn]["k"], "top3_by_layer": tr_[pn]["top3"],
                   "at_l_star": {"rank_all": tr_[pn]["rank_all"][l_star], "rank_within": tr_[pn]["rank_within"][l_star],
                                 "top3": tr_[pn]["top3"][l_star]}} for pn in ("last", "desc_last")}
    s = res["main_last"]
    if s["W"] > s["expected_W"] and s["p_value"] < 0.01:
        reading = "Specific bridge"
    elif s["language_hit_rate"] >= 0.5:
        reading = "Language-group only"
    else:
        reading = "No country signal"
    by_layer = {}
    for grp, ss in (("DESC+DESC2", ("DESC", "DESC2")), ("CAP_TEST", ("CAP_TEST",))):
        ids_ = [r["id"] for r in test_rows if r["set"] in ss and r["k_lang"] and r["k_lang"] >= 2]
        by_layer[grp] = {pn: {"median_rank_within": [statistics.median(test_ranks[i][pn]["rank_within"][l] for i in ids_)
                                                      for l in range(n_layers)],
                              "frac_rank_within_1": [sum(test_ranks[i][pn]["rank_within"][l] == 1 for i in ids_) / len(ids_)
                                                     for l in range(n_layers)],
                              "median_rank_all": [statistics.median(test_ranks[i][pn]["rank_all"][l] for i in ids_)
                                                  for l in range(n_layers)]} for pn in ("last", "desc_last")}
        by_layer[grp]["n"] = len(ids_)
    m.update({"n_classes": len(classes), "test_sets": {k: len(v) for k, v in sets.items()}, "statistic_at_l_star": res,
              "target": target, "by_layer": by_layer,
              "decision": {"reading": reading, "basis": "TARGET + DESC + DESC2, last position, l*",
                           "rule": "W > expected and p < 0.01 -> Specific bridge; else hit rate >= 0.5 -> "
                                   "Language-group only; else No country signal"},
              "note_mexico_capital": "Mexico's probe class is built from prompts naming 'Mexico City', a word cue "
                                     "TARGET does not have (hop/PLAN.md section 7)."})
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({"gate_H3": gate, "statistic": res, "decision": m["decision"],
                      "target_at_lstar": {pn: target[pn]["at_l_star"] for pn in target}}, indent=1))

    # plot: hop/plots.py (circuit/.venv has no matplotlib)

if __name__ == "__main__":
    main()
