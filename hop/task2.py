"""Hop task 2: country features (hop/PLAN.md).

1. Every REF sentence for every country (92 x 10). At the country token (the sentence's last token)
   record every transcoder feature's activation; m_C(l, i) = mean over the 10 sentences,
   n_C(l, i) = number of sentences where it is > 0. Saved to data/cache/hop/ref.npz.
2. Country features of C: n_C >= 8 and m_C >= 2 x max over C' != C of m_C'. Count per country and
   per layer.
3. Bridge set M(item): the country features of the item's bridge country that are active (> 0) in
   the TWO prompt at any intervention position (description's last token and later). Computed for
   every TARGET, DESC and CAP item that is eligible in Task 1.
4. TARGET graph (data/cache/hop/graphs/TARGET.pt, built in Task 0): the 30 feature nodes with the
   highest influence on the " Spanish" logit (library node influence, logit weight one-hot on
   " Spanish"), with layer, position, activation, membership of M(TARGET), top country by m_C and
   its selectivity ratio, and a Neuronpedia link. Graph files exported with the library's
   create_graph_files to results/hop/task2/graph/.
Usage: python hop/task2.py --seed 0
Writes results/hop/task2/{metrics.json, country_feature_counts.csv, bridge_sets.json,
target_top30.csv, graph/, cmd.txt}.
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

NP_URL = "https://www.neuronpedia.org/gemma-2-2b/{layer}-gemmascope-transcoder-16k/{idx}"


def country_features(mC, nC, names):
    """Returns {country: array of (layer, feature)} and the selectivity ratio array."""
    n_c = mC.shape[0]
    out, ratio = {}, np.zeros_like(mC)
    top2 = np.sort(mC, axis=0)[-2:]  # second largest and largest over countries
    for k, c in enumerate(names):
        other_max = np.where(mC[k] >= top2[1], top2[0], top2[1])  # max over the other countries
        ratio[k] = mC[k] / np.maximum(other_max, 1e-12)
        sel = (nC[k] >= 8) & (mC[k] > 0) & (mC[k] >= 2 * other_max)
        out[c] = np.argwhere(sel)
    assert n_c == len(names)
    return out, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task2")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    kept, _ = data.country_table(tok)
    names = sorted(kept)
    t1 = json.load(open(os.path.join(common.RESULTS, "task1", "metrics.json")))
    elig = set(sum(t1["eligible_ids"].values(), []))
    rm = common.load_replacement_model()
    n_layers, n_feat = rm.cfg.n_layers, rm.transcoders.d_transcoder if hasattr(rm.transcoders, "d_transcoder") else None

    ref_path = os.path.join(common.CACHE, "ref.npz")
    if os.path.exists(ref_path):
        z = np.load(ref_path)
        mC, nC = z["mC"], z["nC"]
        assert list(z["names"]) == names
    else:
        mC = nC = None
        for k, c in enumerate(names):
            acc = []
            for s in data.REF:
                ids = torch.tensor([common.tokenize(tok, s.format(C=c))], device=common.DEVICE)
                with torch.no_grad():
                    _, acts = rm.get_activations(ids)
                acc.append(acts[:, -1].float().cpu().numpy())  # [layers, features]
            acc = np.stack(acc)
            if mC is None:
                mC = np.zeros((len(names),) + acc.shape[1:], np.float32)
                nC = np.zeros((len(names),) + acc.shape[1:], np.int8)
            mC[k], nC[k] = acc.mean(0), (acc > 0).sum(0)
            print(k, c, flush=True)
        os.makedirs(common.CACHE, exist_ok=True)
        np.savez(ref_path, mC=mC, nC=nC, names=np.array(names))
    n_layers, n_feat = mC.shape[1], mC.shape[2]
    cf, ratio = country_features(mC.astype(np.float64), nC, names)
    with open(os.path.join(d, "country_feature_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["country", "total"] + [f"L{l}" for l in range(n_layers)])
        for c in names:
            per = np.bincount(cf[c][:, 0], minlength=n_layers) if len(cf[c]) else np.zeros(n_layers, int)
            w.writerow([c, len(cf[c])] + per.tolist())
    np.savez(os.path.join(common.CACHE, "country_features.npz"), **{c: cf[c] for c in names})

    # 3. bridge sets
    its = [it for it in data.items(tok) if it["id"] in elig]
    bridge = {}
    for it in its:
        ids_l = common.tokenize(tok, it["TWO"])
        P = data.two_positions(tok, it, ids_l)
        with torch.no_grad():
            _, acts = rm.get_activations(torch.tensor([ids_l], device=common.DEVICE))
        a = acts.float().cpu().numpy()  # [layers, pos, features]
        ents = []
        for l, i in cf[it["country"]]:
            pos = [p for p in P if a[l, p, i] > 0]
            if pos:
                ents.append({"layer": int(l), "feature": int(i), "positions": pos,
                             "activations": [round(float(a[l, p, i]), 5) for p in pos]})
        bridge[it["id"]] = {"set": it["set"], "country": it["country"], "positions": P,
                            "position_tokens": [tok.decode([ids_l[p]]) for p in P], "M": ents}
        print(it["id"], len(ents), flush=True)
    common.dump(bridge, os.path.join(d, "bridge_sets.json"))

    # 4. TARGET graph
    from circuit_tracer.graph import Graph, compute_node_influence
    g = Graph.from_pt(os.path.join(common.CACHE, "graphs", "TARGET.pt"))
    target = next(it for it in its if it["id"] == "TARGET") if "TARGET" in bridge else data.items(tok)[0]
    toks = [t.vocab_idx for t in g.logit_targets]
    n_feat_nodes = len(g.selected_features)
    rows = []
    in_graph = target["lang_token"] in toks
    if in_graph:
        A = g.adjacency_matrix.double()
        w_ = torch.zeros(A.shape[0], dtype=A.dtype)
        w_[A.shape[0] - len(toks) + toks.index(target["lang_token"])] = 1
        infl = compute_node_influence(A, w_)[:n_feat_nodes]
        top = infl.topk(30)
        Mset = {(e["layer"], e["feature"]): e["positions"] for e in bridge.get("TARGET", {"M": []})["M"]}
        for v, k in zip(top.values, top.indices):
            l, p, i = (int(x) for x in g.active_features[g.selected_features[k]])
            best = int(np.argmax(mC[:, l, i]))
            rows.append({"rank": len(rows) + 1, "influence": round(float(v), 6), "layer": l, "position": p,
                         "token": tok.decode([int(g.input_tokens[p])]), "feature": i,
                         "activation": round(float(g.activation_values[k]), 4),
                         "in_M_TARGET": (l, i) in Mset and p in Mset[(l, i)],
                         "in_M_TARGET_any_position": (l, i) in Mset,
                         "top_country": names[best], "top_country_m": round(float(mC[best, l, i]), 4),
                         "top_country_ratio": round(float(ratio[best, l, i]), 3),
                         "is_country_feature_of_top": bool(((cf[names[best]][:, 0] == l) & (cf[names[best]][:, 1] == i)).any())
                         if len(cf[names[best]]) else False,
                         "neuronpedia": NP_URL.format(layer=l, idx=i)})
        with open(os.path.join(d, "target_top30.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    from circuit_tracer.utils.create_graph_files import create_graph_files
    create_graph_files(g, slug="hop-target-mexico-spanish", output_path=os.path.join(d, "graph"))

    sizes = {s: [len(b["M"]) for b in bridge.values() if b["set"] == s] for s in ("TARGET", "DESC", "CAP")}
    m = {"seed": args.seed, "n_countries": len(names), "n_ref_sentences": len(data.REF),
         "rule": "n_C >= 8 and m_C >= 2 x max over other countries of m_C (and m_C > 0)",
         "country_features_per_country": {c: int(len(cf[c])) for c in names},
         "country_features_total": int(sum(len(cf[c]) for c in names)),
         "countries_with_no_features": [c for c in names if len(cf[c]) == 0],
         "country_features_per_layer": np.bincount(np.concatenate([cf[c][:, 0] for c in names if len(cf[c])]),
                                                   minlength=n_layers).tolist(),
         "M_size": {s: {"n_items": len(v), "n_empty": sum(x == 0 for x in v),
                        "median": statistics.median(v) if v else None, "values": v} for s, v in sizes.items()},
         "M_TARGET": bridge.get("TARGET", {}).get("M"),
         "target_graph": {"spanish_is_logit_target": in_graph,
                          "logit_targets": [tok.decode([t]) for t in toks],
                          "n_feature_nodes": n_feat_nodes, "top30_in_M": sum(r["in_M_TARGET"] for r in rows),
                          "top30_top_country_counts": {c: sum(r["top_country"] == c for r in rows)
                                                       for c in sorted({r["top_country"] for r in rows})}},
         "neuronpedia_url_format": NP_URL,
         "neuronpedia_url_check": "2026-11-19: .../gemma-2-2b/20-gemmascope-transcoder-16k/100 opened; page title "
                                  "'GEMMA-2-2B · 20-GEMMASCOPE-TRANSCODER-16K · 100 | Neuronpedia' (valid feature page)"}
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({k: m[k] for k in ("country_features_total", "countries_with_no_features",
                                          "country_features_per_layer", "target_graph")}, indent=1))
    print({s: {k: v for k, v in x.items() if k != "values"} for s, x in m["M_size"].items()})


if __name__ == "__main__":
    main()
