"""Binding task 3: person units and shared units (binding/PLAN.md).

Per run (its transcoder tc0, or --tc), per layer, per unit u:
  P(u)       trained people for which u is active (a > 0) at the last-name position in START
             (all people for N <= 8k; a fixed random 8,000 (seed 0) for main-32k; runs without
             people (main-0, untrained) use set A people 0..7,999, those of main-8k);
  P_unseen   the same on the 1,000 unseen people (set B 16,000..16,999);
  S(u)       fraction of the 1,000,192 held-out STORY positions where u is active;
  name_share max(share of the most common first name, of the most common last name) over P(u).
DEAD = never active on the held-out STORY positions nor on the FACT prompts (as task 2).
Classes in order: DEAD, STORY (S >= 1e-3), NAME (|P| >= 3, name_share >= 0.8), PERSON
(1 <= |P| <= 10), BIO (other units active on trained people), OTHER.
Null: PERSON count with P computed on a fixed 1,000 of the sampled trained people (seed 0),
against PERSON count with P_unseen on the 1,000 unseen people (same order of rules).
Also: the fraction of sampled trained people with at least one PERSON unit active.
Recurrence data: activations (indices, values) on the first 200,000 held-out STORY positions,
cached in data/cache/binding/ for --recurrence.

  python binding/task3.py --run P2-main-2k-s0 --seed 0
  python binding/task3.py --recurrence
  python binding/task3.py --plots
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as ac  # noqa: E402
from binding import common as B  # noqa: E402
from binding.task2 import Capture, fact_rows  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_SAMPLE = 8000
CACHE = os.path.join(B.ROOT, "data", "cache", "binding")
CLASSES = ["DEAD", "STORY", "NAME", "PERSON", "BIO", "OTHER"]
PAIRS = [("main-8k-s0:0", "main-8k-s0:1", "ceiling: transcoder seed"),
         ("main-8k-s0:0", "main-8k-s1:0", "model seed"),
         ("main-2k-s0:0", "main-32k-s0:0", "fact load"),
         ("main-0-s0:0", "main-32k-s0:0", "no facts vs many"),
         ("main-8k-s0:0", "untrained-s0:0", "null: untrained model")]


def sampled_people(run):
    tp = B.trained_people(run)
    label = "trained"
    if not tp:
        tp = B.people("A")[:N_SAMPLE]
        label = "set A 0..7999 (no trained people)"
    idx = np.arange(len(tp))
    if len(tp) > N_SAMPLE:
        idx = np.sort(np.random.default_rng(0).choice(len(tp), N_SAMPLE, replace=False))
    return [tp[i] for i in idx], idx, label


@torch.no_grad()
def start_active(model, tcs, plist):
    """Per layer: (n_people, k) unit indices and values at the last-name position in START."""
    ids = torch.tensor([B.context_ids("START", *p["name"].split(" ")) for p in plist],
                       device=B.DEVICE)
    out = [[] for _ in range(B.N_LAYER)]
    for i in range(0, len(ids), 4000):
        with Capture(model) as cap:
            model(ids[i:i + 4000])
        for l, tc in enumerate(tcs):
            v, u = tc.encode(cap.hs[l][:, -1])
            out[l].append((u.cpu().numpy(), v.cpu().numpy()))
    return [(np.concatenate([a for a, _ in o]), np.concatenate([b for _, b in o])) for o in out]


def people_sets(units, vals, n_units):
    """List of person-index arrays per unit (active = value > 0)."""
    rows = np.repeat(np.arange(units.shape[0]), units.shape[1])
    u = units.ravel()
    act = vals.ravel() > 0
    rows, u = rows[act], u[act]
    order = np.argsort(u, kind="stable")
    u, rows = u[order], rows[order]
    bounds = np.searchsorted(u, np.arange(n_units + 1))
    return [rows[bounds[i]:bounds[i + 1]] for i in range(n_units)]


def classify(dead, S, Psets, firsts, lasts):
    n = len(S)
    size = np.array([len(p) for p in Psets])
    share = np.zeros(n)
    for i, p in enumerate(Psets):
        if len(p):
            share[i] = max(np.unique(firsts[p], return_counts=True)[1].max(),
                           np.unique(lasts[p], return_counts=True)[1].max()) / len(p)
    cls = np.full(n, "OTHER", dtype=object)
    rest = np.ones(n, dtype=bool)
    for name, rule in [("DEAD", dead), ("STORY", S >= 1e-3),
                       ("NAME", (size >= 3) & (share >= 0.8)),
                       ("PERSON", (size >= 1) & (size <= 10)), ("BIO", size >= 1)]:
        sel = rest & rule
        cls[sel] = name
        rest &= ~sel
    return cls, size, share


@torch.no_grad()
def story_counts(model, tcs, n_units, run, tc_seed):
    """Active counts on the held-out STORY positions and the FACT prompts; caches the first
    200,000 STORY positions' activations for recurrence."""
    rows = B.heldout_story_rows()
    cnt = torch.zeros(B.N_LAYER, n_units, device=B.DEVICE)
    fact_cnt = torch.zeros(B.N_LAYER, n_units, device=B.DEVICE)
    rec = [[] for _ in range(B.N_LAYER)]
    n_rec = 0
    for i in range(0, len(rows), 64):
        with Capture(model) as cap:
            model(torch.from_numpy(rows[i:i + 64]).to(B.DEVICE))
        for l, tc in enumerate(tcs):
            v, u = tc.encode(cap.hs[l].reshape(-1, B.D_MODEL))
            cnt[l].index_add_(0, u[v > 0], torch.ones_like(v[v > 0]))
            if n_rec < B.N_REC_POS:
                take = min(len(v), B.N_REC_POS - n_rec)
                rec[l].append((u[:take].cpu().numpy().astype(np.int32),
                               v[:take].cpu().numpy().astype(np.float32)))
        n_rec += min(64 * 256, max(0, B.N_REC_POS - n_rec))
    fx, fm = fact_rows(B.meta(run))
    for i in range(0, len(fx), 512):
        with Capture(model) as cap:
            model(torch.from_numpy(fx[i:i + 512]).to(B.DEVICE))
        mk = torch.from_numpy(fm[i:i + 512]).to(B.DEVICE)
        for l, tc in enumerate(tcs):
            v, u = tc.encode(cap.hs[l][mk])
            fact_cnt[l].index_add_(0, u[v > 0], torch.ones_like(v[v > 0]))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, f"{run}_tc{tc_seed}_rec.npz"),
             **{f"u{l}": np.concatenate([a for a, _ in rec[l]]) for l in range(B.N_LAYER)},
             **{f"v{l}": np.concatenate([b for _, b in rec[l]]) for l in range(B.N_LAYER)})
    return cnt.cpu().numpy(), fact_cnt.cpu().numpy(), len(rows) * 256


def main_run(args):
    t0 = time.time()
    run = args.run
    tag = run if args.tc == 0 else f"{run}-tc{args.tc}"
    out = B.out_dir("task3", tag)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    model = B.load(run)
    tcs, rec = B.load_transcoders(run, args.tc)
    n_units = rec["n_units"]
    plist, pidx, plabel = sampled_people(run)
    unseen = B.unseen_people()
    sub = np.sort(np.random.default_rng(0).choice(len(plist), 1000, replace=False))
    act_t = start_active(model, tcs, plist)
    act_u = start_active(model, tcs, unseen)
    cnt, fcnt, n_story = story_counts(model, tcs, n_units, run, args.tc)

    def names(pl):
        f = np.array([p["name"].split(" ")[0] for p in pl])
        l = np.array([p["name"].split(" ")[1] for p in pl])
        return f, l

    ft, lt = names(plist)
    fu, lu = names(unseen)
    rows, per_layer = [], []
    for l in range(B.N_LAYER):
        dead = (cnt[l] == 0) & (fcnt[l] == 0)
        S = cnt[l] / n_story
        Pt = people_sets(*act_t[l], n_units)
        Pu = people_sets(*act_u[l], n_units)
        cls, size, share = classify(dead, S, Pt, ft, lt)
        P1k = people_sets(act_t[l][0][sub], act_t[l][1][sub], n_units)
        cls1k, _, _ = classify(dead, S, P1k, ft[sub], lt[sub])
        clsu, size_u, _ = classify(dead, S, Pu, fu, lu)
        person = cls == "PERSON"
        u_t = act_t[l][0]
        has_person = (person[u_t] & (act_t[l][1] > 0)).any(1)
        counts = {c: int((cls == c).sum()) for c in CLASSES}
        per_layer.append({
            "layer": l, "counts": counts,
            "person_on_1000_trained": int((cls1k == "PERSON").sum()),
            "person_on_1000_unseen": int((clsu == "PERSON").sum()),
            "ratio_trained_to_unseen": (int((cls1k == "PERSON").sum())
                                        / max(1, int((clsu == "PERSON").sum()))),
            "frac_people_with_person_unit": float(has_person.mean()),
            "median_S_story_units": float(np.median(S[cls == "STORY"])) if counts["STORY"] else None})
        for u in range(n_units):
            rows.append({"layer": l, "unit": u, "class": cls[u], "P": int(size[u]),
                         "P_unseen": int(size_u[u]), "S": float(S[u]),
                         "name_share": float(share[u])})
        print(f"{time.time() - t0:.0f}s {tag} L{l} {counts} PERSON 1k trained/unseen "
              f"{per_layer[-1]['person_on_1000_trained']}/{per_layer[-1]['person_on_1000_unseen']}"
              f" covered {per_layer[-1]['frac_people_with_person_unit']:.3f}", flush=True)
    with open(os.path.join(out, "units.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res = {"run": run, "tc": args.tc, "recipe": rec["name"], "n_units": n_units,
           "people": plabel, "n_people_sampled": len(plist), "n_unseen": len(unseen),
           "n_story_positions": n_story, "per_layer": per_layer, "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)


def load_rec(spec, l):
    run, tc = spec.split(":")
    with np.load(os.path.join(CACHE, f"{run}_tc{tc}_rec.npz")) as f:
        return f[f"u{l}"], f[f"v{l}"]


def story_units(spec):
    run, tc = spec.split(":")
    tag = run if tc == "0" else f"{run}-tc{tc}"
    with open(os.path.join(B.RES, "task3", tag, "units.csv")) as f:
        R = list(csv.DictReader(f))
    return {l: np.array([int(r["unit"]) for r in R if int(r["layer"]) == l
                         and r["class"] == "STORY"]) for l in range(B.N_LAYER)}


def sparse(u, v, n_units):
    n = u.shape[0]
    m = sp.csr_matrix((v.ravel().astype(np.float64), (np.repeat(np.arange(n), u.shape[1]),
                                                      u.ravel())), shape=(n, n_units))
    m.eliminate_zeros()
    return m


def best_corr(a, b, ua, ub, n_units):
    """For each STORY unit of a (columns ua), the best Pearson correlation with any STORY unit of
    b (columns ub) over the positions."""
    A = sparse(*a, n_units)[:, ua]
    Bm = sparse(*b, n_units)[:, ub]
    n = A.shape[0]
    ma, mb = np.asarray(A.mean(0)).ravel(), np.asarray(Bm.mean(0)).ravel()
    sa = np.sqrt(np.asarray(A.multiply(A).mean(0)).ravel() - ma ** 2)
    sb = np.sqrt(np.asarray(Bm.multiply(Bm).mean(0)).ravel() - mb ** 2)
    best = np.full(len(ua), -np.inf)
    for j0 in range(0, len(ub), 4096):
        Bj = Bm[:, j0:j0 + 4096]
        cov = (A.T @ Bj).toarray() / n - np.outer(ma, mb[j0:j0 + 4096])
        den = np.outer(sa, sb[j0:j0 + 4096])
        c = np.where(den > 0, cov / np.where(den > 0, den, 1), 0)
        best = np.maximum(best, c.max(1))
    return best


def recurrence():
    t0 = time.time()
    res = []
    for a, b, role in PAIRS:
        n_units = 16384
        su_a, su_b = story_units(a), story_units(b)
        for l in range(B.N_LAYER):
            if not len(su_a[l]) or not len(su_b[l]):
                res.append({"pair": f"{a} vs {b}", "role": role, "layer": l, "n_story_a":
                            int(len(su_a[l])), "n_story_b": int(len(su_b[l])), "frac_ge_0.8": None})
                continue
            best = best_corr(load_rec(a, l), load_rec(b, l), su_a[l], su_b[l], n_units)
            res.append({"pair": f"{a} vs {b}", "role": role, "layer": l,
                        "n_story_a": int(len(su_a[l])), "n_story_b": int(len(su_b[l])),
                        "frac_ge_0.8": float((best >= 0.8).mean()),
                        "median_best": float(np.median(best))})
            print(f"{time.time() - t0:.0f}s {role} L{l}: {res[-1]['frac_ge_0.8']:.3f}", flush=True)
    with open(os.path.join(B.out_dir("task3"), "recurrence.json"), "w") as f:
        json.dump({"positions": B.N_REC_POS, "threshold": 0.8, "pairs": res}, f, indent=1)


def plots(runs):
    rows = []
    M = {}
    for tag in runs:
        with open(os.path.join(B.RES, "task3", tag, "metrics.json")) as f:
            M[tag] = json.load(f)
        for pl in M[tag]["per_layer"]:
            rows.append({"run": tag, "layer": pl["layer"], **pl["counts"],
                         "person_on_1000_trained": pl["person_on_1000_trained"],
                         "person_on_1000_unseen": pl["person_on_1000_unseen"],
                         "ratio_trained_to_unseen": pl["ratio_trained_to_unseen"],
                         "frac_people_with_person_unit": pl["frac_people_with_person_unit"]})
    with open(os.path.join(B.out_dir("task3"), "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # PERSON count vs number of people (seed 0 runs), unseen null alongside
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    ns = {"main-2k-s0": 2000, "main-8k-s0": 8000, "main-32k-s0": 32000}
    for key, ax, title in (("person_on_1000_trained", axs[0], "PERSON units (P on 1,000 trained)"),
                           ("person_on_1000_unseen", axs[1], "null: PERSON units (1,000 unseen)")):
        for l in range(B.N_LAYER):
            xs = [n for r, n in ns.items() if r in M]
            ys = [M[r]["per_layer"][l][key] for r in ns if r in M]
            ax.plot(xs, ys, "o-", label=f"L{l}", color=f"C{l}")
            if "ctrl-8k-x4-s0" in M:
                ax.plot([8000], [M["ctrl-8k-x4-s0"]["per_layer"][l][key]], "x", ms=9,
                        color=f"C{l}")
        ax.set_xscale("log")
        ax.set_xlabel("people in the run (x: ctrl-8k-x4)")
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("units")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(B.out_dir("task3"), "person_vs_people.png"), dpi=110)
    plt.close(fig)
    fig, axs = plt.subplots(int(np.ceil(len(M) / 3)), 3, figsize=(14, 3.2 * np.ceil(len(M) / 3)),
                            squeeze=False)
    for ax, tag in zip(axs.ravel(), M):
        bottom = np.zeros(B.N_LAYER)
        for c in CLASSES:
            v = np.array([pl["counts"][c] for pl in M[tag]["per_layer"]])
            ax.bar(range(B.N_LAYER), v, bottom=bottom, label=c)
            bottom += v
        ax.set_title(tag, fontsize=9)
        ax.set_xlabel("layer")
    axs.ravel()[0].legend(fontsize=7)
    for ax in axs.ravel()[len(M):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(B.out_dir("task3"), "classes_by_layer.png"), dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--tc", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recurrence", action="store_true")
    p.add_argument("--plots", nargs="*")
    args = p.parse_args()
    if args.recurrence:
        recurrence()
    elif args.plots is not None:
        plots(args.plots)
    else:
        main_run(args)


if __name__ == "__main__":
    main()
