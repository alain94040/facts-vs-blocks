"""Binding task 4: gate signatures (binding/PLAN.md).

For main-8k seed 0 (and the untrained-model transcoder as the null): 1,000,448 held-out
positions, half STORY (the first 1,954 held-out STORY rows) and half biographies of people 0..999
(1,954 packed rows rendered with a generator independent of the transcoder stream; the untrained
model uses main-8k's people).

Inputs of unit u of layer l at a position: the token embedding; the position embedding; each head
output of layers 0..l (z_h W_out[:, h], 8 per layer); each layer's attention output bias; for
layers m < l, each active transcoder unit's decoder contribution a_v dec_m[v] and the decoder
bias b_dec_m. (The biases are needed for the inputs to add up to the residual; they are listed as
separate inputs.) Error share: |x - sum of inputs| / |x| (the error of the replaced MLPs).

Contribution with the LayerNorm scale fixed at the position: c_v = W_enc[u] . LN_fixed(v) =
(w - mean(w)) . v / sigma, with w = W_enc[u] * ln2.weight and sigma = sqrt(var(x) + eps).
Per active unit and position (u among the position's top-k with a > 0):
  fan-in      smallest number of inputs whose positive contributions reach 80% of the total
              positive contribution;
  AND index   among the 5 largest positive contributors (fewer if fewer are positive), the
              fraction whose removal alone (pre - c_v) drops u below the position's threshold,
              the 33rd-largest pre-activation;
  inhibition  total negative contribution / total absolute contribution.
Averaged over the positions where u is active; units active on >= 100 positions are kept.
Report per layer and task 3 class: median and IQR.

  python binding/task4.py --run main-8k-s0 --seed 0
  python binding/task4.py --run untrained-s0 --people_of main-8k-s0 --seed 0
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

N_ROWS = 1954           # per half: 1,954 x 256 = 500,224 positions
MIN_ACTIVE = 100
SIGS = ["fan_in", "and_index", "inhibition"]


@torch.no_grad()
def batch_signatures(model, tcs, x, acc, err_acc):
    T = x.shape[1]
    caps = {}
    hooks = []
    for l, blk in enumerate(model.blocks):
        hooks.append(blk.ln2.register_forward_hook(
            lambda m, a, o, l=l: caps.__setitem__(("x", l), a[0])))
        hooks.append(blk.ln2.register_forward_hook(
            lambda m, a, o, l=l: caps.__setitem__(("h", l), o)))
        hooks.append(blk.attn.out.register_forward_pre_hook(
            lambda m, a, l=l: caps.__setitem__(("z", l), a[0])))
    model(x)
    for h in hooks:
        h.remove()
    P = x.numel()
    base = [model.tok_emb(x).reshape(P, -1),
            model.pos_emb(torch.arange(T, device=x.device)).repeat(x.shape[0], 1)]
    inputs = list(base)
    for l, blk in enumerate(model.blocks):
        z = caps[("z", l)].reshape(P, ac.N_HEAD, ac.D_HEAD)
        Wo = blk.attn.out.weight.view(B.D_MODEL, ac.N_HEAD, ac.D_HEAD)       # (d, H, dh)
        heads = torch.einsum("phk,dhk->phd", z, Wo)                          # (P, H, d)
        inputs += list(heads.unbind(1)) + [blk.attn.out.bias.expand(P, -1)]
        V = torch.stack(inputs, 1)                                           # (P, n_in, d)
        xl = caps[("x", l)].reshape(P, -1)
        err_acc[l][0] += float((xl - V.sum(1)).norm(dim=1).sum())
        err_acc[l][1] += float(xl.norm(dim=1).sum())
        tc = tcs[l]
        h = caps[("h", l)].reshape(P, -1)
        pre = tc.pre(h)
        thr = pre.topk(tc.k + 1, dim=1).values[:, -1]                       # 33rd largest
        v, u = pre.topk(tc.k, dim=1)
        act = v > 0
        ln = blk.ln2
        sigma = torch.sqrt(xl.var(1, unbiased=False) + ln.eps)               # (P,)
        w = tc.enc[u] * ln.weight                                            # (P, k, d)
        w = w - w.mean(-1, keepdim=True)
        C = torch.einsum("pnd,pkd->pkn", V, w) / sigma[:, None, None]        # (P, k, n_in)
        pos = C.clamp(min=0)
        tot_pos = pos.sum(-1, keepdim=True)
        srt = pos.sort(-1, descending=True).values
        cum = srt.cumsum(-1)
        fan = (cum < 0.8 * tot_pos).sum(-1) + 1
        top5 = srt[..., :5]
        npos5 = (top5 > 0).sum(-1).clamp(min=1)
        need = ((v[..., None] - top5) < thr[:, None, None]) & (top5 > 0)
        andi = need.sum(-1) / npos5
        inh = C.clamp(max=0).abs().sum(-1) / C.abs().sum(-1).clamp(min=1e-12)
        uu = u[act]
        a = acc[l]
        a["n"].index_add_(0, uu, torch.ones_like(uu, dtype=torch.float32))
        a["fan_in"].index_add_(0, uu, fan[act].float())
        a["and_index"].index_add_(0, uu, andi[act].float())
        a["inhibition"].index_add_(0, uu, inh[act].float())
        # this layer's transcoder outputs become inputs of the layers above
        vv, ui = tc.encode(h)
        contrib = vv[..., None] * tc.dec[ui]                                 # (P, k, d)
        inputs += list(contrib.unbind(1)) + [tc.b_dec.expand(P, -1)]
    return P


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--people_of", default=None, help="people for the biography half")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_rows", type=int, default=None, help="smoke runs only")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = B.out_dir("task4", args.run)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    model = B.load(args.run)
    tcs, rec = B.load_transcoders(args.run, 0)
    n = args.max_rows or N_ROWS
    story = B.heldout_story_rows()[:n]
    bio = B.heldout_bio_rows(args.people_of or args.run, n)
    rows = np.concatenate([story, bio])
    acc = [{k: torch.zeros(rec["n_units"], device=B.DEVICE) for k in ["n"] + SIGS}
           for _ in range(B.N_LAYER)]
    err = [[0.0, 0.0] for _ in range(B.N_LAYER)]
    npos = 0
    for i in range(0, len(rows), 4):
        npos += batch_signatures(model, tcs, torch.from_numpy(rows[i:i + 4]).to(B.DEVICE),
                                 acc, err)
        if i % 400 == 0:
            print(f"{time.time() - t0:.0f}s rows {i}/{len(rows)}", flush=True)
    with open(os.path.join(B.RES, "task3", args.run, "units.csv")) as f:
        cls = {(int(r["layer"]), int(r["unit"])): r["class"] for r in csv.DictReader(f)}
    units, summary = [], []
    for l in range(B.N_LAYER):
        a = {k: v.cpu().numpy() for k, v in acc[l].items()}
        keep = np.where(a["n"] >= MIN_ACTIVE)[0]
        for u in keep:
            units.append({"layer": l, "unit": int(u), "class": cls[(l, int(u))],
                          "n_active": int(a["n"][u]),
                          **{s: float(a[s][u] / a["n"][u]) for s in SIGS}})
        for c in ["ALL"] + sorted({x["class"] for x in units if x["layer"] == l}):
            U = [x for x in units if x["layer"] == l and (c == "ALL" or x["class"] == c)]
            row = {"run": args.run, "layer": l, "class": c, "n_units": len(U)}
            for s in SIGS:
                v = np.array([x[s] for x in U])
                row.update({f"{s}_median": float(np.median(v)) if len(v) else None,
                            f"{s}_q25": float(np.quantile(v, 0.25)) if len(v) else None,
                            f"{s}_q75": float(np.quantile(v, 0.75)) if len(v) else None})
            summary.append(row)
    with open(os.path.join(out, "units.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(units[0].keys()))
        w.writeheader()
        w.writerows(units)
    with open(os.path.join(out, "signatures.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    res = {"run": args.run, "positions": npos, "min_active": MIN_ACTIVE,
           "n_units_kept_per_layer": [sum(1 for x in units if x["layer"] == l)
                                      for l in range(B.N_LAYER)],
           "error_share_per_layer": [e / max(t, 1e-12) for e, t in err],
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(units, out)
    print(json.dumps(res), flush=True)


def plot(units, out):
    classes = [c for c in ["STORY", "NAME", "PERSON", "BIO", "OTHER"]
               if any(x["class"] == c for x in units)]
    for s in SIGS:
        fig, ax = plt.subplots(figsize=(12, 4))
        pos, data, labels = [], [], []
        for l in range(B.N_LAYER):
            for ci, c in enumerate(classes):
                v = [x[s] for x in units if x["layer"] == l and x["class"] == c]
                if v:
                    pos.append(l * (len(classes) + 1) + ci)
                    data.append(v)
                    labels.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True)
        for b, lab in zip(bp["boxes"], labels):
            b.set_facecolor(f"C{classes.index(lab)}")
        ax.set_xticks([l * (len(classes) + 1) + (len(classes) - 1) / 2 for l in range(B.N_LAYER)],
                      [f"L{l}" for l in range(B.N_LAYER)])
        ax.set_ylabel(s)
        ax.legend([plt.Rectangle((0, 0), 1, 1, fc=f"C{i}") for i in range(len(classes))],
                  classes, fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{s}.png"), dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
