"""Swap interventions shared by Tasks 3 and 4 (hop/PLAN.md, Task 3 definition).

An intervention is (layer, position, feature, value), passed to
ReplacementModel.feature_intervention with library defaults.
- suppress: at every position in P, each given feature active there (> 0 in the clean run) -> 0;
- inject: at every position in P, each country feature (l, i) of C' whose layer is within +-1 of a
  layer holding a country feature of C -> s x m_C'(l, i).
"""
import json
import os
import random

import common
import numpy as np
import torch


class Context:
    def __init__(self, tok, rm):
        self.tok, self.rm = tok, rm
        z = np.load(os.path.join(common.CACHE, "ref.npz"))
        self.names = list(z["names"])
        self.mC = z["mC"].astype(np.float64)
        self.idx = {c: k for k, c in enumerate(self.names)}
        cf = np.load(os.path.join(common.CACHE, "country_features.npz"))
        self.cf = {c: cf[c].reshape(-1, 2) for c in self.names}
        # features active at REF: > 0 in at least one REF sentence of some country
        self.ref_active = z["nC"].max(0) > 0  # [layers, features]
        self.ref_mean = self.mC.mean(0)  # mean over countries of m_C
        self.cf_set = {c: {(int(l), int(i)) for l, i in self.cf[c]} for c in self.names}

    def ids(self, text):
        import common as c
        return torch.tensor([c.tokenize(self.tok, text)], device=common.DEVICE)

    def acts(self, ids):
        with torch.no_grad():
            _, a = self.rm.get_activations(ids)
        return a.float().cpu().numpy()

    def run(self, ids, ivs):
        """Final-position logits (float64, CPU) with the interventions applied."""
        with torch.no_grad():
            if ivs:
                logits, _ = self.rm.feature_intervention(ids, ivs)
            else:
                logits = self.rm(ids)
        return logits[0, -1].cpu().double()

    # building blocks
    def suppress(self, acts, feats, P):
        return [(int(l), p, int(i), 0.0) for l, i in feats for p in P if acts[l, p, i] > 0]

    def inject_feats(self, C, C2, s):
        """[(layer, feature, value)] of C' to inject for source C."""
        layers = {int(l) for l, _ in self.cf[C]}
        ok = {l + dl for l in layers for dl in (-1, 0, 1)}
        k = self.idx[C2]
        return [(int(l), int(i), s * float(self.mC[k, l, i])) for l, i in self.cf[C2] if int(l) in ok]

    def random_inject_feats(self, real, C, C2, rng):
        """Same count per layer as `real`, random features active at REF (not country features of
        C or C'), values proportional to their mean REF activation, scaled to the same total."""
        excl = self.cf_set[C] | self.cf_set[C2]
        out = []
        for l in sorted({l for l, _, _ in real}):
            n = sum(1 for x in real if x[0] == l)
            cand = [int(i) for i in np.flatnonzero(self.ref_active[l]) if (l, int(i)) not in excl]
            out += [(l, i) for i in rng.sample(cand, n)]
        tot = sum(v for _, _, v in real)
        w = np.array([self.ref_mean[l, i] for l, i in out])
        w = w / w.sum() * tot if w.sum() > 0 else np.full(len(out), tot / max(len(out), 1))
        return [(l, i, float(v)) for (l, i), v in zip(out, w)]

    @staticmethod
    def place(feats, P):
        return [(l, p, i, v) for l, i, v in feats for p in P]


def swap_countries(items, kept, hop2_ok, seed=0):
    """3 swap countries per item, seed 0, among countries whose language differs from the bridge
    country's and whose own HOP2 is top-1 correct; TARGET also gets Brazil (fixed, 4th)."""
    rng = random.Random(seed)
    out = {}
    for it in items:
        C = it["country"]
        cand = sorted(c for c in hop2_ok if kept[c]["language"] != kept[C]["language"]
                      and not (it["id"] == "TARGET" and c == "Brazil"))
        out[it["id"]] = rng.sample(cand, 3) + (["Brazil"] if it["id"] == "TARGET" else [])
    return out


def load_swaps():
    return json.load(open(os.path.join(common.RESULTS, "task3", "swaps.json")))
