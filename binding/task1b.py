"""Binding task 1b: where a known person is told apart (binding/PLAN.md, Revision 1).

END prompts (recall prefixes, Phase 2 tokenisation):
  CITY      <eot>F L was born in            (birth_city template 1)  7 tokens, END = 'in'
  EMPLOYER  <eot>The employer of F L is     (employer template 4)    9 tokens, END = 'is'

1. Interaction at END: Task 1 steps 1 and 4 (IR by point, real - control with bootstrap CI,
   P_bind_END; binding neurons) at the END position of both prompts, same quadruples as Task 1.
   Supplementary (not used by any decision row): the same at the answer-space position, i.e. the
   prompt followed by the bare space token that begins every answer (contexts CITY_SP, EMPLOYER_SP).

2. Causal tracing (Meng et al. 2022) on people 0..199 of the run.
   Scored token: every answer starts with the bare space token 'G' (id of ' '), identical for all
   people, so "the person's correct first answer token" is taken as the first token of the
   attribute value, i.e. the answer token after the space. It is read at the answer-space
   position (prompt + space appended), where the model predicts it. Patched positions: first
   name .. END as in the plan, plus that answer-space position (reported as column 'SP').
   - corrupted: F1 L1 replaced by F2 L2 of the person's real quadruple (Task 1 quadruples; people
     0..199 without one in the 500 get an F2 L2 drawn with the same rule from seed + 1).
   - residual: clean residual written in at one (position, point); MLP-only: clean MLP output of
     layer l written in at one position, rest corrupted.
   - score = (p_restored - p_corrupt) / (p_clean - p_corrupt), averaged over items with
     p_clean - p_corrupt >= 0.1.
   - control: clean = a second unknown pair F3 L3 (drawn with the same rule, seed + 2),
     corrupted = F2 L2; target token = the clean run's argmax at the scored position; reported
     as p_restored - p_corrupt (all 200 items), next to the real p_restored - p_corrupt (all 200).

  python binding/task1b.py --run P2-main-2k-s0 --seed 0
  python binding/task1b.py --summarise <runs...>
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
from binding import common as B  # noqa: E402
from binding.task1 import diff_ci, interaction, ir, p_bind  # noqa: E402
from train import evaluate as ev  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROMPTS = {"CITY": ("birth_city", 0), "EMPLOYER": ("employer", 3)}
CONTEXTS_1B = ["CITY", "EMPLOYER", "CITY_SP", "EMPLOYER_SP"]
N_TRACE = 200
MIN_GAP = 0.1


def prompt_ids(ctx, f, l):
    t = B.tok_ids()
    enc = lambda s: t["tok"].encode(s).ids  # noqa: E731
    fi, li = B.name_ids(f, l)
    name = [fi, t["space"], li]
    base = ctx.replace("_SP", "")
    if base == "CITY":
        ids = [t["eot"]] + name + enc(" was born in")
    else:
        ids = [t["eot"]] + enc("The employer of") + [t["space"]] + name + enc(" is")
    if ctx.endswith("_SP"):
        ids = ids + [t["space"]]
    return ids


def first_name_pos(ctx):
    return 1 if ctx.startswith("CITY") else 5


def check_tokenisation(run):
    """The hand-built prompts equal Phase 2's recall prompts, and the answer starts with space."""
    tp = B.trained_people(run)[:50] or B.people("A")[:50]
    pr = {(a, t, i): (p, ans) for a, t, p, ans, i in ev.build_prompts(list(enumerate(tp)))}
    sp = B.tok_ids()["space"]
    for i, p in enumerate(tp):
        f, l = p["name"].split(" ")
        for ctx, (attr, ti) in PROMPTS.items():
            ref, ans = pr[(attr, ti, i)]
            assert prompt_ids(ctx, f, l) == ref, (ctx, p["name"])
            assert ans[0] == sp and len(ans) >= 3, (ctx, p["name"], ans)
    return True


# ---------------------------------------------------------------- step 1


def measure_ctx(model, quads, ctx, mlp_acts=True):
    ids = []
    for f1, l1, f2, l2 in quads:
        ids += [prompt_ids(ctx, f1, l1), prompt_ids(ctx, f1, l2),
                prompt_ids(ctx, f2, l1), prompt_ids(ctx, f2, l2)]
    ids = torch.tensor(ids, device=B.DEVICE)
    rs, acts = [], []
    for i in range(0, len(ids), 2000):
        r, a = B.points_at_last(model, ids[i:i + 2000], mlp_acts=True)
        rs.append(r.cpu().double())
        acts.append(a.cpu().double())
    r = torch.cat(rs).view(len(quads), 4, len(B.POINTS), -1)
    a = torch.cat(acts).view(len(quads), 4, B.N_LAYER, -1)
    return r, a


def step1(model, quads, seed):
    out, rows = {}, []
    for ctx in CONTEXTS_1B:
        r_real, a_real = measure_ctx(model, quads["real"], ctx)
        r_ctrl, a_ctrl = measure_ctx(model, quads["control"], ctx)
        IRr, IRc = ir(r_real), ir(r_ctrl)
        d, lo, hi = diff_ci(IRr, IRc, seed)
        pb = p_bind(d, lo, hi)
        ir_n = interaction(a_real).abs().numpy()
        ic_n = interaction(a_ctrl).abs().numpy()
        thr = np.percentile(ic_n, 99, axis=0)
        breal, bctrl = ir_n > thr[None], ic_n > thr[None]
        neur = []
        for l in range(B.N_LAYER):
            per = breal[:, l].sum(0)
            neur.append({"layer": l,
                         "mean_binding_per_real_quadruple": float(breal[:, l].sum(1).mean()),
                         "mean_binding_per_control_quadruple": float(bctrl[:, l].sum(1).mean()),
                         "n_distinct_binding_neurons": int((per > 0).sum()),
                         "max_people_per_neuron": int(per.max()),
                         "max_people_frac": float(per.max() / len(quads["real"])),
                         "hist_people_per_neuron": np.bincount(
                             per, minlength=len(quads["real"]) + 1).tolist()})
        out[ctx] = {"supplementary": ctx.endswith("_SP"),
                    "median_IR_real": np.median(IRr, 0).tolist(),
                    "median_IR_control": np.median(IRc, 0).tolist(),
                    "diff": d.tolist(), "diff_lo": lo.tolist(), "diff_hi": hi.tolist(),
                    "P_bind": pb, "neurons": neur}
        for k, pt in enumerate(B.POINTS):
            rows.append({"context": ctx, "point": pt, "median_IR_real": float(np.median(IRr[:, k])),
                         "median_IR_control": float(np.median(IRc[:, k])), "diff": d[k],
                         "diff_lo": lo[k], "diff_hi": hi[k], "is_P_bind": pt == pb})
        print(f"  {ctx}: P_bind={pb} diff={np.round(d, 3).tolist()}", flush=True)
    return out, rows


# ---------------------------------------------------------------- step 2


def unknown_pair(rng, f1, l1, names, avoid=()):
    P = B.pools()
    fn, ln = P["first_names"], P["last_names"]
    while True:
        f2, l2 = fn[rng.integers(len(fn))], ln[rng.integers(len(ln))]
        if f2 == f1 or l2 == l1 or (f2, l2) in avoid:
            continue
        if {f"{f1} {l2}", f"{f2} {l1}", f"{f2} {l2}"} & names:
            continue
        return f2, l2


def trace_items(run, quads, seed):
    names = B.all_person_names()
    by_person = {(q[0], q[1]): (q[2], q[3]) for q in quads["real"]}
    rng1, rng2 = np.random.default_rng(seed + 1), np.random.default_rng(seed + 2)
    items, n_from_quads = [], 0
    for p in B.trained_people(run)[:N_TRACE]:
        f1, l1 = p["name"].split(" ")
        if (f1, l1) in by_person:
            f2, l2 = by_person[(f1, l1)]
            n_from_quads += 1
        else:
            f2, l2 = unknown_pair(rng1, f1, l1, names)
        f3, l3 = unknown_pair(rng2, f2, l2, names | {p["name"]}, avoid={(f1, l1)})
        items.append({"person": p, "clean": (f1, l1), "corrupt": (f2, l2), "ctrl_clean": (f3, l3)})
    return items, n_from_quads


@torch.no_grad()
def forward_all(model, ids):
    """Residual at every position for the 13 points (B, T, 13, d) and MLP outputs (B, T, 6, d)."""
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1], device=ids.device))
    res, mlp = [x], []
    for blk in model.blocks:
        x = x + blk.attn(blk.ln1(x))
        res.append(x)
        m = blk.mlp(blk.ln2(x))
        mlp.append(m)
        x = x + m
        res.append(x)
    probs = torch.softmax(model.head(model.ln_f(x[:, -1])), -1)
    return torch.stack(res, 2), torch.stack(mlp, 2), probs


@torch.no_grad()
def forward_patch(model, ids, pos, kind, idx, val):
    """ids (N, T); row r gets val[r] written at position pos[r]: the residual after point idx
    (kind 'res') or the MLP output of layer idx (kind 'mlp'). Returns probs at the last position."""
    rows = torch.arange(ids.shape[0], device=ids.device)
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1], device=ids.device))

    def put(t, i):
        if kind == "res" and idx == i:
            t = t.clone()
            t[rows, pos] = val
        return t
    x = put(x, 0)
    for l, blk in enumerate(model.blocks):
        x = put(x + blk.attn(blk.ln1(x)), 2 * l + 1)
        m = blk.mlp(blk.ln2(x))
        if kind == "mlp" and idx == l:
            m = m.clone()
            m[rows, pos] = val
        x = put(x + m, 2 * l + 2)
    return torch.softmax(model.head(model.ln_f(x[:, -1])), -1)


def trace(model, clean_ids, corrupt_ids, target, positions):
    """Returns p_clean (n,), p_corrupt (n,), p_res (n_pos, 13, n), p_mlp (n_pos, 6, n)."""
    n = clean_ids.shape[0]
    res_c, mlp_c, pr_c = forward_all(model, clean_ids)
    _, _, pr_x = forward_all(model, corrupt_ids)
    ar = torch.arange(n, device=B.DEVICE)
    p_clean, p_corrupt = pr_c[ar, target], pr_x[ar, target]
    npos = len(positions)
    ids_rep = corrupt_ids.repeat(npos, 1)
    pos_rep = torch.tensor(positions, device=B.DEVICE).repeat_interleave(n)
    tgt_rep = target.repeat(npos)
    ar_rep = torch.arange(n * npos, device=B.DEVICE)
    item_rep = ar.repeat(npos)
    p_res = torch.zeros(npos, len(B.POINTS), n)
    p_mlp = torch.zeros(npos, B.N_LAYER, n)
    for k in range(len(B.POINTS)):
        val = res_c[item_rep, pos_rep, k]
        pr = forward_patch(model, ids_rep, pos_rep, "res", k, val)
        p_res[:, k] = pr[ar_rep, tgt_rep].view(npos, n).cpu()
    for l in range(B.N_LAYER):
        val = mlp_c[item_rep, pos_rep, l]
        pr = forward_patch(model, ids_rep, pos_rep, "mlp", l, val)
        p_mlp[:, l] = pr[ar_rep, tgt_rep].view(npos, n).cpu()
    return p_clean.cpu(), p_corrupt.cpu(), p_res, p_mlp, pr_c


def step2(model, run, items):
    out, rows = {}, []
    for ctx in PROMPTS:
        sctx = ctx + "_SP"
        T = len(prompt_ids(sctx, *items[0]["clean"]))
        positions = list(range(first_name_pos(ctx), T))
        labels = [f"{i}:{B.tok_ids()['tok'].id_to_token(t)}"
                  for i, t in zip(positions, prompt_ids(sctx, *items[0]["clean"])[positions[0]:])]
        labels[-1] = f"{positions[-1]}:SP"
        attr = PROMPTS[ctx][0]
        sp = B.tok_ids()["space"]
        target = []
        for it in items:
            pr = ev.build_prompts([(0, it["person"])])
            ans = [a for a, t, p, aa, _ in pr if a == attr and t == PROMPTS[ctx][1]]
            full = [aa for a, t, p, aa, _ in pr if a == attr and t == PROMPTS[ctx][1]][0]
            assert full[0] == sp and ans
            target.append(full[1])
        target = torch.tensor(target, device=B.DEVICE)
        mk = lambda key: torch.tensor([prompt_ids(sctx, *it[key]) for it in items],  # noqa: E731
                                      device=B.DEVICE)
        clean, corrupt, cclean = mk("clean"), mk("corrupt"), mk("ctrl_clean")
        pc, px, p_res, p_mlp, _ = trace(model, clean, corrupt, target, positions)
        pr_cc = forward_all(model, cclean)[2]
        ctgt = pr_cc.argmax(-1)
        qc, qx, q_res, q_mlp, _ = trace(model, cclean, corrupt, ctgt, positions)
        gap = pc - px
        keep = gap >= MIN_GAP
        nk = int(keep.sum())
        C = {"positions": positions, "labels": labels, "n_items": len(items),
             "n_kept": nk, "p_clean_mean": float(pc.mean()), "p_corrupt_mean": float(px.mean()),
             "control_p_clean_mean": float(qc.mean()), "control_p_corrupt_mean": float(qx.mean()),
             "control_target_is_space": float((ctgt == sp).float().mean())}
        for var, pr_, qr_ in (("residual", p_res, q_res), ("mlp", p_mlp, q_mlp)):
            score = ((pr_[..., keep] - px[keep]) / gap[keep]).mean(-1) if nk else \
                torch.full(pr_.shape[:2], float("nan"))
            dp_real = (pr_ - px).mean(-1)
            dp_ctrl = (qr_ - qx).mean(-1)
            C[var] = {"score": score.tolist(), "dp_real": dp_real.tolist(),
                      "dp_control": dp_ctrl.tolist()}
            cols = B.POINTS if var == "residual" else [f"mlp{l}" for l in range(B.N_LAYER)]
            for i, lab in enumerate(labels):
                for k, c in enumerate(cols):
                    rows.append({"run": run, "prompt": ctx, "variant": var, "position": lab,
                                 "point": c, "score": float(score[i, k]),
                                 "dp_real": float(dp_real[i, k]),
                                 "dp_control": float(dp_ctrl[i, k])})
        out[ctx] = C
        print(f"  trace {ctx}: kept {nk}/{len(items)} p_clean {C['p_clean_mean']:.3f} "
              f"p_corrupt {C['p_corrupt_mean']:.3f}", flush=True)
    return out, rows


def heatmaps(res, run, out):
    for ctx, C in res["trace"].items():
        for var in ("residual", "mlp"):
            cols = B.POINTS if var == "residual" else [f"mlp{l}" for l in range(B.N_LAYER)]
            fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
            for ax, key, title in zip(axs, ["score", "dp_real", "dp_control"],
                                      [f"score (n={C['n_kept']})", "real p_restored - p_corrupt",
                                       "control p_restored - p_corrupt"]):
                M = np.array(C[var][key])
                im = ax.imshow(M, aspect="auto", cmap="viridis",
                               vmin=min(0, np.nanmin(M)), vmax=max(1e-3, np.nanmax(M)))
                ax.set_yticks(range(len(C["labels"])), C["labels"], fontsize=7)
                ax.set_xticks(range(len(cols)), cols, fontsize=7, rotation=90)
                ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax)
            fig.suptitle(f"{run} {ctx} {var}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"trace_{ctx}_{var}.png"), dpi=110)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, ctx in enumerate(CONTEXTS_1B):
        C = res["interaction"][ctx]
        ax.plot(C["median_IR_real"], "-", color=f"C{i}", label=f"{ctx} real")
        ax.plot(C["median_IR_control"], "--", color=f"C{i}", lw=0.8)
    ax.set_xticks(range(len(B.POINTS)), B.POINTS)
    ax.set_ylabel("median IR (solid real, dashed control)")
    ax.legend(fontsize=7)
    ax.set_title(f"{run}: interaction at END")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ir_END.png"), dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = B.out_dir("task1b", args.run)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    model = B.load(args.run)
    check_tokenisation(args.run)
    quads = B.quadruples(args.run, seed=args.seed)
    items, n_q = trace_items(args.run, quads, args.seed)
    with torch.no_grad():  # checks: the forward reproduces the model; a no-op patch changes nothing
        ids = torch.tensor([prompt_ids("CITY_SP", *it["clean"]) for it in items[:50]],
                           device=B.DEVICE)
        res_c, mlp_c, pr = forward_all(model, ids)
        ref = torch.softmax(model(ids)[0][:, -1], -1)
        fwd_err = float((pr - ref).abs().max())
        pos = torch.full((50,), 3, device=B.DEVICE)
        pp = forward_patch(model, ids, pos, "res", 5, res_c[:, 3, 5])
        pm = forward_patch(model, ids, pos, "mlp", 2, mlp_c[:, 3, 2])
        noop_err = float(max((pp - pr).abs().max(), (pm - pr).abs().max()))
    assert fwd_err < 1e-4 and noop_err < 1e-4, (fwd_err, noop_err)
    print(f"{time.time() - t0:.0f}s checks ok (fwd {fwd_err:.1e}, noop {noop_err:.1e})", flush=True)
    inter, rows1 = step1(model, quads, args.seed)
    tr, rows2 = step2(model, args.run, items)
    res = {"run": args.run, "seed": args.seed, "points": B.POINTS,
           "forward_check_max_abs_prob_err": fwd_err, "noop_patch_max_abs_prob_err": noop_err,
           "n_trace_items": len(items), "n_trace_corrupt_from_task1_quadruples": n_q,
           "P_bind_END": {c: inter[c]["P_bind"] for c in PROMPTS},
           "interaction": inter, "trace": tr, "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    for name, rows in (("interaction_rows.csv", rows1), ("trace_rows.csv", rows2)):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    heatmaps(res, args.run, out)
    print(f"{time.time() - t0:.0f}s done P_bind_END={res['P_bind_END']}", flush=True)


def summarise(runs):
    """summary.csv: per run and prompt, P_bind_END, max real - control at END, and the MLP-only
    tracing location of the largest score (decision rows)."""
    rows = []
    for run in runs:
        with open(os.path.join(B.RES, "task1b", run, "metrics.json")) as f:
            m = json.load(f)
        for ctx in PROMPTS:
            I = m["interaction"][ctx]
            C = m["trace"][ctx]
            fn = first_name_pos(ctx)
            name_pos = [fn, fn + 1, fn + 2]
            end_pos = C["positions"][-2]
            out = {"run": run, "prompt": ctx, "P_bind_END": I["P_bind"],
                   "max_diff_END": float(np.nanmax(np.array(I["diff"], dtype=float))),
                   "max_diff_point": B.POINTS[int(np.nanargmax(np.array(I["diff"], dtype=float)))],
                   "max_diff_lo": I["diff_lo"][int(np.nanargmax(np.array(I["diff"], dtype=float)))],
                   "n_kept": C["n_kept"], "p_clean": C["p_clean_mean"],
                   "p_corrupt": C["p_corrupt_mean"]}
            for var in ("residual", "mlp"):
                S = np.array(C[var]["score"])
                i, k = np.unravel_index(np.nanargmax(S), S.shape)
                pos = C["positions"][i]
                out[f"{var}_max_score"] = float(S[i, k])
                out[f"{var}_max_at"] = f"{C['labels'][i]} / " + (
                    B.POINTS[k] if var == "residual" else f"mlp{k}")
                if var == "mlp":
                    name_max = float(np.nanmax(S[[C["positions"].index(q) for q in name_pos]]))
                    end_row = S[C["positions"].index(end_pos)]
                    out["mlp_max_at_END"] = bool(pos == end_pos)
                    out["mlp_max_layer"] = int(k)
                    out["mlp_max_at_last_name"] = bool(pos == fn + 2)
                    out["mlp_max_at_SP"] = bool(pos == C["positions"][-1])
                    out["mlp_name_max"] = name_max
                    out["mlp_END_max"] = float(np.nanmax(end_row))
                    out["mlp_END_max_layer"] = int(np.nanargmax(end_row))
            rows.append(out)
    d = B.out_dir("task1b")
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    allrows = []
    for run in runs:
        for name in ("interaction_rows.csv",):
            with open(os.path.join(B.RES, "task1b", run, name)) as f:
                allrows += [{"run": run, **r} for r in csv.DictReader(f)]
    with open(os.path.join(d, "interaction_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(allrows)
    for ctx in PROMPTS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, run in enumerate(runs):
            R = [r for r in allrows if r["run"] == run and r["context"] == ctx]
            ax.plot([float(r["median_IR_real"]) for r in R], "-", color=f"C{i}", label=run)
            ax.plot([float(r["median_IR_control"]) for r in R], "--", color=f"C{i}", lw=0.8)
        ax.set_xticks(range(len(B.POINTS)), B.POINTS)
        ax.set_ylabel("median IR at END (solid real, dashed control)")
        ax.set_title(ctx)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(d, f"ir_END_{ctx}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarise":
        summarise(sys.argv[2:])
    else:
        main()
