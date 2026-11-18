"""Binding task 1: where a person is assembled (binding/PLAN.md).

Per run and context (START, MID):
1. IR = |I| / D at the 13 points for 500 real and 500 control quadruples. Medians, and the
   difference median(real IR) - median(control IR) with a bootstrap 95% CI (2,000 resamples, real
   and control resampled independently). P_bind = first point with difference >= 0.1 and CI
   excluding 0.
2. Routing heads: mean attention from the last-name to the first-name position in START over the
   run's people 0..999; heads with mean >= 0.3 are mean-ablated together (Phase 2 means) and step
   1 is redone (real and control quadruples both under the ablation); the difference at P_bind and
   m5 is reported ablated vs not.
3. If P_bind is an MLP point mL: all 1,024 neurons of MLP l mean-ablated, same numbers, plus FACT
   recall on people 0..999.
4. Neuron interactions i_n (post-GELU at the last-name position); binding for a real quadruple
   if |i_n| > 99th percentile of |i_n| over the 500 control quadruples (per neuron and context).

  python binding/task1.py --run P2-main-2k-s0 --seed 0
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import common as ac  # noqa: E402
from binding import common as B  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

N_BOOT = 2000
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def prompts(quads, ctx):
    """4 prompts per quadruple: F1L1, F1L2, F2L1, F2L2."""
    out = []
    for f1, l1, f2, l2 in quads:
        out += [B.context_ids(ctx, f1, l1), B.context_ids(ctx, f1, l2),
                B.context_ids(ctx, f2, l1), B.context_ids(ctx, f2, l2)]
    return torch.tensor(out, device=B.DEVICE)


@torch.no_grad()
def measure(model, quads, ctx, hooks=None, mlp_acts=False):
    ids = prompts(quads, ctx)
    rs, acts = [], []
    for i in range(0, len(ids), 4000):
        o = B.points_at_last(model, ids[i:i + 4000], mlp_acts=mlp_acts)
        if mlp_acts:
            rs.append(o[0].cpu().double())
            acts.append(o[1].cpu().double())
        else:
            rs.append(o.cpu().double())
    r = torch.cat(rs).view(len(quads), 4, len(B.POINTS), -1)
    a = torch.cat(acts).view(len(quads), 4, B.N_LAYER, -1) if mlp_acts else None
    return r, a


def interaction(x):
    """x: (Q, 4, ...) values for F1L1, F1L2, F2L1, F2L2 -> I (Q, ...)."""
    return x[:, 0] - x[:, 1] - x[:, 2] + x[:, 3]


def ir(r):
    """r: (Q, 4, P, d) -> IR (Q, P)."""
    I = interaction(r).norm(dim=-1)
    D = torch.stack([(r[:, a] - r[:, b]).norm(dim=-1) for a, b in PAIRS]).mean(0)
    return (I / D).numpy()


def diff_ci(real, ctrl, seed=0):
    """median(real) - median(control) per point, bootstrap CI. real, ctrl: (Q, P)."""
    rng = np.random.default_rng(seed)
    d = np.median(real, 0) - np.median(ctrl, 0)
    bs = np.empty((N_BOOT, real.shape[1]))
    for b in range(N_BOOT):
        bs[b] = (np.median(real[rng.integers(0, len(real), len(real))], 0)
                 - np.median(ctrl[rng.integers(0, len(ctrl), len(ctrl))], 0))
    return d, np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def p_bind(d, lo, hi):
    for k, p in enumerate(B.POINTS):
        if d[k] >= 0.1 and (lo[k] > 0 or hi[k] < 0):
            return p
    return None


@torch.no_grad()
def routing_attention(model, run):
    """Mean attention (layer, head) from the last-name position (3) to the first-name position
    (1) in START, over the run's people 0..999 (or 1,000 pool names for runs without people)."""
    tp = B.trained_people(run)[:1000]
    names = [p["name"].split(" ") for p in tp]
    ids = torch.tensor([B.context_ids("START", f, l) for f, l in names], device=B.DEVICE)
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(4, device=B.DEVICE))
    out = np.zeros((B.N_LAYER, ac.N_HEAD))
    for l, blk in enumerate(model.blocks):
        h = blk.ln1(x)
        q, k, _ = blk.attn.qkv(h).split(B.D_MODEL, dim=2)
        q = q.view(len(ids), 4, ac.N_HEAD, ac.D_HEAD).transpose(1, 2)
        k = k.view(len(ids), 4, ac.N_HEAD, ac.D_HEAD).transpose(1, 2)
        s = (q[:, :, 3:4] @ k.transpose(-1, -2)) / ac.D_HEAD ** 0.5      # (n, H, 1, 4)
        out[l] = torch.softmax(s, -1)[:, :, 0, 1].mean(0).cpu().numpy()
        x = blk(x)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = B.out_dir("task1", args.run)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    model = B.load(args.run)
    meta = B.meta(args.run)
    means, n_ref = ac.component_means(model)
    quads = B.quadruples(args.run, seed=args.seed)
    with torch.no_grad():  # check: the hand-written forward reproduces the model's logits
        ids = prompts(quads["real"][:50], "MID")
        r = B.points_at_last(model, ids)
        ref = model(ids)[0][:, -1]
        fwd_err = float((model.head(model.ln_f(r[:, -1])) - ref).abs().max())
    res = {"run": args.run, "seed": args.seed, "n_quadruples": {k: len(v) for k, v in quads.items()},
           "points": B.POINTS, "forward_check_max_abs_logit_err": fwd_err,
           "contexts": {}}
    att = routing_attention(model, args.run)
    routing = [(int(l), int(h)) for l, h in zip(*np.where(att >= 0.3))]
    res["routing_attention"] = att.tolist()
    res["routing_heads"] = routing
    ab_heads = {}
    for l, h in routing:
        ab_heads.setdefault(l, []).append(h)
    rows = []
    for ctx in B.CONTEXTS:
        r_real, a_real = measure(model, quads["real"], ctx, mlp_acts=True)
        r_ctrl, a_ctrl = measure(model, quads["control"], ctx, mlp_acts=True)
        IRr, IRc = ir(r_real), ir(r_ctrl)
        d, lo, hi = diff_ci(IRr, IRc, args.seed)
        pb = p_bind(d, lo, hi)
        C = {"median_IR_real": np.median(IRr, 0).tolist(),
             "median_IR_control": np.median(IRc, 0).tolist(),
             "diff": d.tolist(), "diff_lo": lo.tolist(), "diff_hi": hi.tolist(), "P_bind": pb}
        for k, pt in enumerate(B.POINTS):
            rows.append({"run": args.run, "context": ctx, "point": pt,
                         "median_IR_real": C["median_IR_real"][k],
                         "median_IR_control": C["median_IR_control"][k],
                         "diff": d[k], "diff_lo": lo[k], "diff_hi": hi[k],
                         "is_P_bind": pt == pb})
        at = [k for k in ([pb] if pb else []) + ["m5"]]
        idx = [B.POINTS.index(x) for x in at]

        def ablated(neurons=None, heads=None):
            with ac.Hooks(model, means=means, ablate_neurons=neurons, ablate_heads=heads):
                rr, _ = measure(model, quads["real"], ctx)
                rc, _ = measure(model, quads["control"], ctx)
            dd, l2, h2 = diff_ci(ir(rr), ir(rc), args.seed)
            dr, _, _ = diff_ci(ir(rr), IRc, args.seed)
            return {x: {"diff_ablated": float(dd[i]), "ci": [float(l2[i]), float(h2[i])],
                        "diff_ablated_vs_unablated_control": float(dr[i]),
                        "diff_unablated": float(d[i])} for x, i in zip(at, idx)}

        C["routing_ablation"] = ablated(heads=ab_heads) if routing else None
        if pb and pb.startswith("m"):
            l = int(pb[1:])
            nm = {l: torch.arange(B.D_MLP)}
            C["mlp_ablation"] = {"layer": l, **ablated(neurons=nm)}
            with ac.Hooks(model, means=means, ablate_neurons=nm):
                C["mlp_ablation"]["fact_recall_ablated"] = ac.fact_recall(model, meta)[0]
            C["mlp_ablation"]["fact_recall_unablated"] = ac.fact_recall(model, meta)[0]
        else:
            C["mlp_ablation"] = None
        # neurons
        ir_n = interaction(a_real).abs().numpy()       # (Q, 6, 1024)
        ic_n = interaction(a_ctrl).abs().numpy()
        thr = np.percentile(ic_n, 99, axis=0)           # (6, 1024)
        breal = ir_n > thr[None]
        bctrl = ic_n > thr[None]
        neur = []
        for l in range(B.N_LAYER):
            per_neuron = breal[:, l].sum(0)
            neur.append({"layer": l,
                         "mean_binding_per_real_quadruple": float(breal[:, l].sum(1).mean()),
                         "mean_binding_per_control_quadruple": float(bctrl[:, l].sum(1).mean()),
                         "n_distinct_binding_neurons": int((per_neuron > 0).sum()),
                         "max_people_per_neuron": int(per_neuron.max()),
                         "max_people_frac": float(per_neuron.max() / len(quads["real"])),
                         "hist_people_per_neuron": np.bincount(
                             per_neuron, minlength=len(quads["real"]) + 1).tolist()})
        C["neurons"] = neur
        res["contexts"][ctx] = C
        print(f"{time.time() - t0:.0f}s {args.run} {ctx}: P_bind={pb} "
              f"diff={np.round(d, 3).tolist()}", flush=True)
    res["n_reference_positions"] = n_ref
    res["wall_seconds"] = time.time() - t0
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out, "summary_rows.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarise(runs):
    """Merge per-run rows into task1/summary.csv and draw one plot per context."""
    rows = []
    for run in runs:
        p = os.path.join(B.RES, "task1", run, "summary_rows.csv")
        if os.path.exists(p):
            with open(p) as f:
                rows += list(csv.DictReader(f))
    d = B.out_dir("task1")
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for ctx in B.CONTEXTS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, run in enumerate(runs):
            R = [r for r in rows if r["run"] == run and r["context"] == ctx]
            if not R:
                continue
            x = range(len(R))
            ax.plot(x, [float(r["median_IR_real"]) for r in R], "-", color=f"C{i}", label=run)
            ax.plot(x, [float(r["median_IR_control"]) for r in R], "--", color=f"C{i}", lw=0.8)
        ax.set_xticks(range(len(B.POINTS)), B.POINTS)
        ax.set_xlabel("point (last-name position)")
        ax.set_ylabel("median IR (solid: real, dashed: control)")
        ax.set_title(f"{ctx} context")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"ir_{ctx}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarise":
        summarise(sys.argv[2:])
    else:
        main()
