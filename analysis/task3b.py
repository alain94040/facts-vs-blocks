"""Task 3b: where the per-person content is stored (PLAN.md Phase 2b).

Reuses STORY, FACT, FORMAT, the mean reference sample and I_D from Phase 2 (analysis/common.py)
and the Task 3 rankings and knowledge sets (results/task3/<run>/).

  1. layer-matched random baseline for the knowledge set (3 draws, seed 0)
  2. per-layer curves: top k of layer L by I_FACT, k = 8 .. 1024; K_L
  3. per-person attribution I_p (people 0-199, 24 FACT prompts each), participation ratio
  4. own-person ablation (people 0-49; top 32 by I_p, all layers and layers 1-5 only)
  5. linear read-out probes (people 0-999, 5 attributes, 7 layer points, NAME and END positions)
  6. probe transfer between forks (only when both forks are analysed in one call)
  7. fork knowledge set: mean activation on entity / other FORMAT / STORY tokens

  python analysis/task3b.py --runs main-2k-s0 ... --task3 results/task3 --out results/task3b
  (main-0-s0: steps 5 only)
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_RANDOM = 3
KS_LAYER = [8 * 2 ** i for i in range(8)]  # 8 .. 1024
N_ATTR_PEOPLE = 200
N_OWN = 50
OTHER_POOL = (50, 250)
N_OTHER = 49
TOP_OWN = 32
PR_KEEP = 0.2
PR_LOW = 10
PROBE_ATTRS = ["birth_city", "university", "major", "employer", "employer_city"]
N_PROBE, N_PROBE_TRAIN = 1000, 800
PROBE_STEPS, PROBE_LR, PROBE_WD = 500, 1e-2, 1e-3
N_POINTS = C.N_LAYER + 1
MEASURES = ["recall", "format_loss", "name_copy", "story_loss"]


def nobio(meta):
    return not meta.get("n_people")


def load_task3(task3_dir, run):
    with open(os.path.join(task3_dir, run, "metrics.json")) as f:
        m = json.load(f)
    imp = np.load(os.path.join(task3_dir, run, "importance.npz"))
    return m, imp["neurons_FACT"]


# ---------------------------------------------------------------- steps 1 and 2

def layer_matched(model, means, meta, t3, rng):
    ks = t3["knowledge_set"]
    if not ks["K"]:
        return None
    out = []
    for d in range(N_RANDOM):
        flat = np.concatenate([l * C.D_MLP + rng.choice(C.D_MLP, n, replace=False)
                               for l, n in enumerate(ks["neurons_per_layer"]) if n])
        with C.Hooks(model, means, ablate_neurons=C.neuron_sets(flat)):
            out.append(C.evaluate_all(model, meta))
    return {"K": ks["K"], "neurons_per_layer": ks["neurons_per_layer"],
            "knowledge_set": {m: ks["at_K"][m] for m in MEASURES},
            "layer_matched_random": out,
            "layer_matched_random_mean": {m: float(np.mean([r[m] for r in out]))
                                          for m in MEASURES}}


def per_layer_curves(model, means, meta, imp_fact, base):
    res = {}
    for l in range(C.N_LAYER):
        order = np.argsort(-imp_fact[l], kind="stable")
        rows = []
        for k in KS_LAYER:
            with C.Hooks(model, means, ablate_neurons={l: torch.from_numpy(order[:k]).long()}):
                rows.append({"k": k, **C.evaluate_all(model, meta)})
        KL = next((r["k"] for r in rows if r["recall"] <= 0.5 * base["recall"]), None)
        res[l] = {"curve": rows, "K_L": KL}
    return res


# ---------------------------------------------------------------- steps 3 and 4

def person_prompts(meta, lo, hi):
    """FACT prompts of people lo..hi-1 (templates 1-4), grouped: list of (person, [prompts])."""
    pr = C.fact_prompts(meta)
    by = {}
    for q in pr:
        if lo <= q[4] < hi:
            by.setdefault(q[4], []).append(q)
    return [(p, by[p]) for p in range(lo, hi)]


def batches(prompts, batch=1024):
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        L = max(len(p) + len(a) for _, _, p, a, _ in chunk) - 1
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        y = torch.full((len(chunk), L), -1, dtype=torch.long)
        for r, (_, _, p, a, _) in enumerate(chunk):
            seq = p + a
            x[r, :len(seq) - 1] = torch.tensor(seq[:-1])
            y[r, len(p) - 1:len(seq) - 1] = torch.tensor(a)
        yield i, x, y


def per_person_importance(model, means, groups):
    """I_p for every person and neuron: (n_people, 6, 1024). Lbar_p = mean loss over the person's
    answer tokens; contributions summed over the person's rows (rows are independent)."""
    flat = [q for _, qs in groups for q in qs]
    owner = np.array([gi for gi, (_, qs) in enumerate(groups) for _ in qs])
    n_ans = np.array([sum(len(q[3]) for q in qs) for _, qs in groups], dtype=np.float64)
    out = np.zeros((len(groups), C.N_LAYER, C.D_MLP))
    mu = [m.to(C.DEVICE) for m in means["mlp"]]
    for i0, x, y in batches(flat):
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        own = owner[i0:i0 + len(x)]
        w = torch.from_numpy(1.0 / n_ans[own]).float().to(C.DEVICE)
        model.zero_grad(set_to_none=True)
        with C.Hooks(model, capture=True) as hk:
            logits, _ = model(x)
            tl = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1).clamp(min=0),
                                 reduction="none").view(y.shape) * (y >= 0)
            (tl.sum(1) * w).sum().backward()
        last = torch.where(y >= 0, torch.arange(y.shape[1], device=C.DEVICE), -1).max(1).values
        vm = (torch.arange(y.shape[1], device=C.DEVICE)[None, :] <= last[:, None])[..., None]
        for l in range(C.N_LAYER):
            a = hk.acts[("mlp", l)]
            c = ((mu[l] - a.detach()) * a.grad * vm).sum(1).cpu().double().numpy()  # (rows, 1024)
            np.add.at(out[:, l], own, c)
    model.zero_grad(set_to_none=True)
    return np.abs(out)


@torch.no_grad()
def prompt_correct(model, prompts):
    ok = np.zeros(len(prompts), dtype=bool)
    for i0, x, y in batches(prompts):
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)
        logits, _ = model(x)
        m = y >= 0
        ok[i0:i0 + len(x)] = ((logits.argmax(-1) == y) | ~m).all(1).cpu().numpy()
    return ok


def participation(Ip):
    s = Ip.sum(0)  # (6, 1024)
    pr = s ** 2 / np.maximum((Ip ** 2).sum(0), 1e-300)
    flat = s.ravel()
    keep = flat >= np.quantile(flat, 1 - PR_KEEP)
    keep = keep.reshape(s.shape)
    by_layer = []
    for l in range(C.N_LAYER):
        v = pr[l][keep[l]]
        by_layer.append({"layer": l, "n_kept": int(keep[l].sum()),
                         "median_PR": float(np.median(v)) if len(v) else None,
                         "n_PR_below_10": int((v < PR_LOW).sum())})
    return pr, keep, by_layer


def own_person(model, means, Ip, groups, rng):
    """People 0..49: ablate their top-32 neurons by I_p; own vs other recall."""
    pool = [g for g in groups if OTHER_POOL[0] <= g[0] < OTHER_POOL[1]]
    own_groups = [g for g in groups if g[0] < N_OWN]
    others = [rng.choice(len(pool), N_OTHER, replace=False) for _ in own_groups]
    all_prompts = [q for _, qs in own_groups + pool for q in qs]
    slices, i = {}, 0
    for p, qs in own_groups + pool:
        slices[p] = slice(i, i + len(qs))
        i += len(qs)
    base = prompt_correct(model, all_prompts)
    res = {}
    for variant, layers in [("all_layers", range(C.N_LAYER)), ("layers_1_5", range(1, C.N_LAYER))]:
        rows = []
        for gi, (p, _) in enumerate(own_groups):
            score = Ip[gi].copy()
            mask = np.zeros(C.N_LAYER, dtype=bool)
            mask[list(layers)] = True
            score[~mask] = -np.inf
            top = np.argsort(-score.ravel(), kind="stable")[:TOP_OWN]
            with C.Hooks(model, means, ablate_neurons=C.neuron_sets(top)):
                ok = prompt_correct(model, all_prompts)
            oth = [pool[j][0] for j in others[gi]]
            own_b, own_a = base[slices[p]].mean(), ok[slices[p]].mean()
            oth_b = np.mean([base[slices[q]].mean() for q in oth])
            oth_a = np.mean([ok[slices[q]].mean() for q in oth])
            rows.append({"person": p, "own_base": float(own_b), "own_ablated": float(own_a),
                         "own_drop": float(own_b - own_a), "other_base": float(oth_b),
                         "other_ablated": float(oth_a), "other_drop": float(oth_b - oth_a),
                         "layers_of_top32": np.bincount(top // C.D_MLP,
                                                        minlength=C.N_LAYER).tolist()})
        res[variant] = {"mean_own_drop": float(np.mean([r["own_drop"] for r in rows])),
                        "mean_other_drop": float(np.mean([r["other_drop"] for r in rows])),
                        "people": rows}
    return res


# ---------------------------------------------------------------- steps 5 and 6

_people = {}


def people(s):
    if s not in _people:
        with open(os.path.join(C.ROOT, "data", "cache", "bios", f"people_{s}.json")) as f:
            _people[s] = json.load(f)
    return _people[s]


def classes(attr):
    """Class list = every value of the attribute in sets A and B (the full pool)."""
    return sorted({p[attr] for s in "AB" for p in people(s)})


@torch.no_grad()
def residuals(model, pset):
    """Residual stream at NAME (last-name token) and END (last prompt token) of the template 1
    FACT prompt, people 0..999 of set pset: {attr: {"NAME"|"END": (7, 1000, 256)}}."""
    tok = C.ev.tokenizer()
    vocab = tok.get_vocab()
    ppl = people(pset)[:N_PROBE]
    prompts = C.ev.build_prompts(list(enumerate(ppl)))
    out = {}
    for attr in PROBE_ATTRS:
        pr = [q for q in prompts if q[0] == attr and q[1] == 0]
        assert [q[4] for q in pr] == list(range(N_PROBE))
        name_pos = [q[2].index(vocab[ppl[q[4]]["name"].split(" ")[1]]) for q in pr]
        end_pos = [len(q[2]) - 1 for q in pr]
        L = max(len(q[2]) for q in pr)
        x = torch.zeros((len(pr), L), dtype=torch.long)
        for r, q in enumerate(pr):
            x[r, :len(q[2])] = torch.tensor(q[2])
        acts = []
        hs = [model.blocks[0].register_forward_pre_hook(lambda m, a: acts.append(a[0]))]
        hs += [b.register_forward_hook(lambda m, a, o: acts.append(o)) for b in model.blocks]
        model(x.to(C.DEVICE))
        for h in hs:
            h.remove()
        R = torch.stack(acts).cpu()  # (7, n, T, 256)
        idx = torch.arange(len(pr))
        out[attr] = {"NAME": R[:, idx, torch.tensor(name_pos)].numpy(),
                     "END": R[:, idx, torch.tensor(end_pos)].numpy()}
    return out


def fit_probe(X, y, n_cls, seed):
    torch.manual_seed(seed)
    X = torch.from_numpy(X).float()
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lin = torch.nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(lin.parameters(), lr=PROBE_LR, weight_decay=PROBE_WD)
    Xs, yt = (X - mu) / sd, torch.from_numpy(y).long()
    for _ in range(PROBE_STEPS):
        opt.zero_grad()
        F.cross_entropy(lin(Xs), yt).backward()
        opt.step()
    return lin, mu, sd


@torch.no_grad()
def probe_acc(probe, X, y):
    lin, mu, sd = probe
    pred = lin((torch.from_numpy(X).float() - mu) / sd).argmax(-1).numpy()
    return float((pred == y).mean())


def probes(R, pset, seed):
    res, fitted = {}, {}
    ppl = people(pset)[:N_PROBE]
    for attr in PROBE_ATTRS:
        cls = classes(attr)
        y = np.array([cls.index(p[attr]) for p in ppl])
        res[attr] = {"n_classes": len(cls), "chance": 1 / len(cls), "NAME": [], "END": []}
        for pos in ["NAME", "END"]:
            for pt in range(N_POINTS):
                X = R[attr][pos][pt]
                pb = fit_probe(X[:N_PROBE_TRAIN], y[:N_PROBE_TRAIN], len(cls), seed)
                fitted[(attr, pos, pt)] = pb
                res[attr][pos].append({"point": pt,
                                       "train_acc": probe_acc(pb, X[:N_PROBE_TRAIN],
                                                              y[:N_PROBE_TRAIN]),
                                       "test_acc": probe_acc(pb, X[N_PROBE_TRAIN:],
                                                             y[N_PROBE_TRAIN:])})
    return res, fitted


def transfer(fitted_src, R_tgt, tgt_set):
    ppl = people(tgt_set)[:N_PROBE]
    res = {}
    for attr in PROBE_ATTRS:
        cls = classes(attr)
        y = np.array([cls.index(p[attr]) for p in ppl])[N_PROBE_TRAIN:]
        res[attr] = {pos: [probe_acc(fitted_src[(attr, pos, pt)],
                                     R_tgt[attr][pos][pt][N_PROBE_TRAIN:], y)
                           for pt in range(N_POINTS)] for pos in ["NAME", "END"]}
    return res


# ---------------------------------------------------------------- step 7

@torch.no_grad()
def knowledge_activation(model, neurons, rng):
    """Mean activation of the given flat neuron ids (layer 0) on entity tokens and other tokens of
    FORMAT, and on STORY tokens; same for 8 random layer-0 neurons."""
    rand = rng.choice(C.D_MLP, len(neurons), replace=False)
    sets = {"knowledge_set": np.asarray(neurons), "random_layer0": rand}
    assert all(n < C.D_MLP for n in neurons)
    sums = {k: {"entity": 0.0, "format_other": 0.0, "story": 0.0} for k in sets}
    per = {k: {"entity": np.zeros(len(v)), "format_other": np.zeros(len(v)),
               "story": np.zeros(len(v))} for k, v in sets.items()}
    cnt = {"entity": 0, "format_other": 0, "story": 0}
    with C.Hooks(model, capture=True) as hk:
        rows = C.story_rows()[:, :-1]
        for i in range(0, len(rows), 128):
            model(torch.from_numpy(rows[i:i + 128]).to(C.DEVICE))
            a = hk.acts[("mlp", 0)].reshape(-1, C.D_MLP).cpu().numpy()
            for k, v in sets.items():
                per[k]["story"] += a[:, v].sum(0)
            cnt["story"] += len(a)
        for ids, _ in C.format_bios():
            x = torch.tensor(ids)[None].to(C.DEVICE)
            model(x)
            a = hk.acts[("mlp", 0)][0].cpu().numpy()
            ent = np.array(ids) >= C.ev.BASE_VOCAB
            for k, v in sets.items():
                per[k]["entity"] += a[ent][:, v].sum(0)
                per[k]["format_other"] += a[~ent][:, v].sum(0)
            cnt["entity"] += int(ent.sum())
            cnt["format_other"] += int((~ent).sum())
    out = {"token_counts": cnt}
    for k, v in sets.items():
        pm = {t: (per[k][t] / cnt[t]).tolist() for t in cnt}
        out[k] = {"neurons": [int(n) for n in v], "per_neuron_mean": pm,
                  "mean": {t: float(np.mean(pm[t])) for t in cnt}}
    return out


# ---------------------------------------------------------------- driver

def analyse(run, task3_dir, out_root, seed):
    t0 = time.time()
    model, meta = C.load_model(run), C.run_meta(run)
    pset = C.fact_set(meta)
    res = {"run": run, "seed": seed, "ckpt": C.ckpt_path(run), "fact_set": pset,
           "n_people": meta.get("n_people", 0)}
    fitted, R = None, None
    if not nobio(meta):
        t3, imp_fact = load_task3(task3_dir, run)
        means, _ = C.component_means(model)
        base = C.evaluate_all(model, meta)
        res["base"] = base
        res["layer_matched"] = layer_matched(model, means, meta, t3, np.random.default_rng(seed))
        print(run, "step 1", round(time.time() - t0), "s", flush=True)
        curves = per_layer_curves(model, means, meta, imp_fact, base)
        res["per_layer"] = {str(l): v for l, v in curves.items()}
        res["K_L"] = [curves[l]["K_L"] for l in range(C.N_LAYER)]
        print(run, "step 2 K_L", res["K_L"], round(time.time() - t0), "s", flush=True)
        groups = person_prompts(meta, 0, OTHER_POOL[1])
        Ip = per_person_importance(model, means, groups[:N_ATTR_PEOPLE])
        pr, keep, by_layer = participation(Ip)
        res["participation"] = {"by_layer": by_layer, "n_people": N_ATTR_PEOPLE,
                                "keep_top_frac": PR_KEEP,
                                "pr_kept_by_layer": [pr[l][keep[l]].tolist()
                                                     for l in range(C.N_LAYER)]}
        print(run, "step 3", [(b["median_PR"], b["n_PR_below_10"]) for b in by_layer],
              flush=True)
        res["own_person"] = own_person(model, means, Ip, groups, np.random.default_rng(seed))
        print(run, "step 4", {k: (round(v["mean_own_drop"], 3), round(v["mean_other_drop"], 3))
                              for k, v in res["own_person"].items()}, flush=True)
        if t3["knowledge_set"]["K"] and (run.startswith("fork") or run in C.PILOT):
            kset = np.argsort(-imp_fact.ravel(), kind="stable")[:t3["knowledge_set"]["K"]]
            l0 = [int(n) for n in kset if n < C.D_MLP][:8]
            res["knowledge_activation"] = knowledge_activation(model, l0,
                                                               np.random.default_rng(seed))
    R = residuals(model, pset)
    res["probes"], fitted = probes(R, pset, seed)
    print(run, "step 5", {a: max(r["test_acc"] for r in v["END"])
                          for a, v in res["probes"].items()}, round(time.time() - t0), "s",
          flush=True)
    res["wall_seconds"] = time.time() - t0
    out = os.path.join(out_root, run)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    return res, fitted, R


def summary(out_root, all_res):
    rows = []
    for run, r in all_res.items():
        row = {"run": run, "n_people": r["n_people"]}
        if "base" in r:
            lm = r["layer_matched"]
            row.update({"base_recall": r["base"]["recall"], "K": lm["K"] if lm else None})
            if lm:
                for m in MEASURES:
                    row[f"kset_{m}"] = lm["knowledge_set"][m]
                    row[f"lmrand_{m}"] = lm["layer_matched_random_mean"][m]
            for l in range(C.N_LAYER):
                row[f"K_L{l}"] = r["K_L"][l]
            for b in r["participation"]["by_layer"]:
                row[f"medPR_L{b['layer']}"] = b["median_PR"]
                row[f"nPR<10_L{b['layer']}"] = b["n_PR_below_10"]
            for v, d in r["own_person"].items():
                row[f"own_drop_{v}"] = d["mean_own_drop"]
                row[f"other_drop_{v}"] = d["mean_other_drop"]
        for a, v in r["probes"].items():
            row[f"probe_{a}_END_best"] = max(x["test_acc"] for x in v["END"])
            row[f"probe_{a}_NAME_best"] = max(x["test_acc"] for x in v["NAME"])
        rows.append(row)
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(os.path.join(out_root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plots(out_root, all_res):
    bio = {k: v for k, v in all_res.items() if "K_L" in v}
    if bio:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (run, r) in enumerate(bio.items()):
            y = [k if k else 2048 for k in r["K_L"]]
            ax.plot(range(C.N_LAYER), y, "o-", label=run)
        ax.axhline(2048, color="k", ls=":", lw=0.8)
        ax.set_yscale("log", base=2)
        ax.set_xlabel("layer")
        ax.set_ylabel("K_L (2048 = recall never halves)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "K_L.png"), dpi=110)
        plt.close(fig)

        hist_runs = [r for r in ["main-2k-s0", "main-8k-s0", "main-32k-s0"] if r in bio] or \
            list(bio)[:3]
        fig, axs = plt.subplots(len(hist_runs), C.N_LAYER, figsize=(18, 3 * len(hist_runs)),
                                squeeze=False)
        for i, run in enumerate(hist_runs):
            for l in range(C.N_LAYER):
                v = bio[run]["participation"]["pr_kept_by_layer"][l]
                axs[i, l].hist(v, bins=np.linspace(1, N_ATTR_PEOPLE, 40))
                axs[i, l].set_title(f"{run} L{l} (n={len(v)})", fontsize=8)
        fig.supxlabel("participation ratio over 200 people (kept neurons)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_root, "PR_hist.png"), dpi=100)
        plt.close(fig)

    fig, axs = plt.subplots(1, len(PROBE_ATTRS), figsize=(20, 4))
    for ax, attr in zip(axs, PROBE_ATTRS):
        for i, (run, r) in enumerate(all_res.items()):
            p = r["probes"][attr]
            ax.plot(range(N_POINTS), [x["test_acc"] for x in p["END"]], "-o", color=f"C{i % 10}",
                    ms=3, label=f"{run} END")
            ax.plot(range(N_POINTS), [x["test_acc"] for x in p["NAME"]], ":", color=f"C{i % 10}",
                    label=f"{run} NAME")
        ax.axhline(r["probes"][attr]["chance"], color="k", lw=0.8)
        ax.set_title(attr)
        ax.set_xlabel("layer point (0 = embeddings)")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("probe test accuracy (people 800-999)")
    axs[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "probes.png"), dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--task3", default="results/task3")
    p.add_argument("--out", default="results/task3b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transfer", nargs=2, default=None,
                   help="two runs (e.g. fork-A-s0 fork-B-s0) for probe transfer")
    args = p.parse_args()
    out_root = os.path.join(C.ROOT, args.out)
    task3 = os.path.join(C.ROOT, args.task3)
    os.makedirs(out_root, exist_ok=True)
    all_res, fitted, Rs = {}, {}, {}
    for run in args.runs:
        all_res[run], fitted[run], Rs[run] = analyse(run, task3, out_root, args.seed)
    if args.transfer:
        a, b = args.transfer
        tr = {f"{a} probes on {b}": transfer(fitted[a], Rs[b], C.fact_set(C.run_meta(b))),
              f"{b} probes on {a}": transfer(fitted[b], Rs[a], C.fact_set(C.run_meta(a)))}
        with open(os.path.join(out_root, "probe_transfer.json"), "w") as f:
            json.dump(tr, f, indent=2)
        for k, v in tr.items():
            print("transfer", k, {at: max(x["END"]) for at, x in v.items()}, flush=True)
    summary(out_root, all_res)
    plots(out_root, all_res)
    with open(os.path.join(out_root, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")


if __name__ == "__main__":
    main()
