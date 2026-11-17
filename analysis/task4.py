"""Task 4: swapping parts between models (PLAN.md Phase 2).

1. Weight interpolation (alpha = 0, .25, .5, .75, 1), STORY loss at each point, between A and B and
   between A and the unrelated model. Barrier = highest loss on the path - mean of the two ends.
2. Hybrids, base A with parts from B and the mirror image, plus the control (H1, H2 with the
   unrelated model as donor, for both bases). "Attention" = qkv and out weights and biases of each
   block; "MLP" = up and down weights and biases. Embeddings and LayerNorms stay with the base.
   H3 = the MLP layers holding 80% of the base model's knowledge set (Task 3): layers taken in
   decreasing order of knowledge-set neurons until they cover at least 80% of it.
   Recorded per hybrid: recall on the base model's people and on the donor's people (FACT, first
   1,000 people of each set), FORMAT loss, STORY loss (and name-copy, fact loss).

  python analysis/task4.py --a fork-A-s0 --b fork-B-s0 --unrelated main-8k-s1 --task3 results/task3 \
      --out results/task4 --seed 0
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BARRIER_FLAG = 0.1


def attn_keys(l):
    return [f"blocks.{l}.attn.{m}.{p}" for m in ("qkv", "out") for p in ("weight", "bias")]


def mlp_keys(l):
    return [f"blocks.{l}.mlp.{m}.{p}" for m in ("up", "down") for p in ("weight", "bias")]


def state(run):
    return {k: v.detach().clone() for k, v in C.load_model(run, "cpu").state_dict().items()}


def with_state(model, sd):
    model.load_state_dict(sd)
    return model.eval()


def interpolate(model, sa, sb, meta):
    out = []
    for a in ALPHAS:
        sd = {k: (1 - a) * sa[k] + a * sb[k] for k in sa}
        out.append({"alpha": a, "story_loss": C.mean_loss(with_state(model, sd), "STORY", meta)})
    losses = [r["story_loss"] for r in out]
    return {"path": out, "barrier": max(losses) - 0.5 * (losses[0] + losses[-1])}


def h3_layers(task3_dir, run):
    p = os.path.join(task3_dir, run, "metrics.json")
    if not os.path.exists(p):
        return None, "no Task 3 result for " + run
    with open(p) as f:
        ks = json.load(f)["knowledge_set"]
    if not ks["K"]:
        return None, "Task 3 found no knowledge set (recall never halved)"
    cnt = np.array(ks["neurons_per_layer"])
    layers, cum = [], 0
    for l in np.argsort(-cnt, kind="stable"):
        layers.append(int(l))
        cum += cnt[l]
        if cum >= 0.8 * cnt.sum():
            break
    return sorted(layers), None


def hybrids_for(task3_dir, base):
    n = C.N_LAYER
    hy = [("H1", "all attention", [k for l in range(n) for k in attn_keys(l)]),
          ("H2", "all MLP", [k for l in range(n) for k in mlp_keys(l)])]
    layers, why = h3_layers(task3_dir, base)
    hy.append(("H3", f"MLP layers {layers} (80% of base knowledge set)" if layers else why,
               [k for l in layers for k in mlp_keys(l)] if layers else None))
    for l in range(n):
        hy.append((f"H{4 + l}", f"MLP layer {l}", mlp_keys(l)))
    hy.append(("H10", "token embeddings (tied with the output layer)", ["tok_emb.weight", "head.weight"]))
    return hy


def evaluate(model, base_meta, donor_meta):
    r = C.evaluate_all(model, base_meta)
    r["recall_base_people"] = r.pop("recall")
    r["fact_loss_base_people"] = r.pop("fact_loss")
    r["recall_donor_people"], r["fact_loss_donor_people"] = C.fact_recall(model, donor_meta)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default="fork-A-s0")
    p.add_argument("--b", default="fork-B-s0")
    p.add_argument("--unrelated", default="main-8k-s1")
    p.add_argument("--task3", default="results/task3")
    p.add_argument("--out", default="results/task4")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = os.path.join(C.ROOT, args.out)
    task3 = os.path.join(C.ROOT, args.task3)
    os.makedirs(out, exist_ok=True)
    cmd = "PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n"
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write(cmd)

    runs = [args.a, args.b, args.unrelated]
    sd = {r: state(r) for r in runs}
    meta = {r: C.run_meta(r) for r in runs}
    model = C.load_model(args.a)

    interp = {f"{args.a}->{args.b}": interpolate(model, sd[args.a], sd[args.b], meta[args.a]),
              f"{args.a}->{args.unrelated}": interpolate(model, sd[args.a], sd[args.unrelated],
                                                        meta[args.a])}
    fork_barrier = interp[f"{args.a}->{args.b}"]["barrier"]
    interp_res = {"alphas": ALPHAS, "paths": interp, "fork_barrier": fork_barrier,
                  "fork_barrier_above_0.1": fork_barrier > BARRIER_FLAG, "seed": args.seed}
    with open(os.path.join(out, "interpolation.json"), "w") as f:
        json.dump(interp_res, f, indent=2)
    print("interpolation", json.dumps({k: v["barrier"] for k, v in interp.items()}), flush=True)

    swaps = {"fork_barrier": fork_barrier, "fork_barrier_above_0.1": fork_barrier > BARRIER_FLAG,
             "originals": {}, "hybrids": [], "seed": args.seed}
    for r in runs:
        other = args.b if r == args.a else args.a
        swaps["originals"][r] = evaluate(with_state(model, sd[r]), meta[r], meta[other])
        swaps["originals"][r]["donor_people_from"] = other
    pairs = [(args.a, args.b), (args.b, args.a)]
    for base, donor in pairs:
        for hid, desc, keys in hybrids_for(task3, base):
            row = {"id": hid, "base": base, "donor": donor, "parts": desc, "control": False}
            if keys is None:
                row["skipped"] = True
            else:
                hyb = dict(sd[base])
                hyb.update({k: sd[donor][k] for k in keys})
                row.update(evaluate(with_state(model, hyb), meta[base], meta[donor]))
            swaps["hybrids"].append(row)
            print(base, hid, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)},
                  flush=True)
        for hid, desc, keys in hybrids_for(task3, base)[:2]:  # control: unrelated donor
            hyb = dict(sd[base])
            hyb.update({k: sd[args.unrelated][k] for k in keys})
            row = {"id": hid + "-ctrl", "base": base, "donor": args.unrelated, "parts": desc,
                   "control": True,
                   **evaluate(with_state(model, hyb), meta[base], meta[args.unrelated])}
            swaps["hybrids"].append(row)
            print(base, row["id"], {k: round(v, 3) for k, v in row.items()
                                    if isinstance(v, float)}, flush=True)
    swaps["wall_seconds"] = time.time() - t0
    with open(os.path.join(out, "swaps.json"), "w") as f:
        json.dump(swaps, f, indent=2)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"interpolation": interp_res, "swaps": swaps}, f, indent=2)
    plot(swaps, out, pairs)
    print("done", round(time.time() - t0), "s")


def plot(swaps, out, pairs):
    fig, axs = plt.subplots(len(pairs), 3, figsize=(17, 4 * len(pairs)), squeeze=False)
    for row_i, (base, _) in enumerate(pairs):
        rows = [h for h in swaps["hybrids"] if h["base"] == base and not h.get("skipped")]
        orig = swaps["originals"][base]
        labels = ["orig"] + [h["id"] for h in rows]
        x = np.arange(len(labels))
        for ax, (key, title) in zip(axs[row_i], [
                (("recall_base_people", "recall_donor_people"), "recall: base / donor people"),
                (("story_loss",), "STORY loss"), (("format_loss",), "FORMAT loss")]):
            w = 0.8 / len(key)
            for j, k in enumerate(key):
                ax.bar(x + j * w, [orig[k]] + [h[k] for h in rows], w, label=k)
            ax.set_xticks(x + 0.4 - w / 2)
            ax.set_xticklabels(labels, rotation=60, fontsize=8)
            ax.set_title(f"base {base}: {title}")
            if len(key) > 1:
                ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "swaps.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
