"""Middle task 1: per-layer statistics of input and update (middle/PLAN.md, Part 1).

For every layer l, with x the residual stream entering it and y the one leaving it (fp16 model on
MPS), at positions 8..255 of each chunk:
  z = [x / rms(x), 1]      rms(x) = sqrt(mean(x^2) + eps), eps of the layer's input RMSNorm,
                           no gain (dimension d + 1)
  u = y - x                (dimension d)
Accumulated sums per split (FIT_A = chunks 0-189, FIT_B = 190-379, TEST = 380-479; FIT is
FIT_A + FIT_B): Szz = sum z z^T, Szu = sum z u^T, Suu = sum u u^T, n. Each batch's products are
formed in float32 on MPS and added to float64 accumulators on CPU. No per-token activations are
stored. Layers are processed in groups small enough for memory; every group is one pass over the
chunks, and its sums are written to data/cache/middle/<model>/L<l>_<split>.npz when it finishes.

  python middle/task1.py --model dev --seed 0
  python middle/task1.py --model dev --seed 0 --max_chunks 16 --smoke    (short timing run)
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
from middle import common as C  # noqa: E402
from repeat import flow as FL  # noqa: E402

MEM_BUDGET = 8e9  # bytes of float64 accumulators held at once


def split_of(i):
    return next(s for s, r in C.SPLITS.items() if i in r)


@torch.no_grad()
def run(flow, ids, group, batch, log):
    d = flow.cfg.hidden_size
    acc = {(l, s): {"Szz": np.zeros((d + 1, d + 1)), "Szu": np.zeros((d + 1, d)),
                    "Suu": np.zeros((d, d)), "n": np.zeros(())}
           for l in group for s in C.SPLITS}
    last = max(group)
    t0, ntok = time.time(), 0
    # batches never straddle a split boundary
    order = [(s, list(r)) for s, r in C.SPLITS.items()]
    for s, idx in order:
        for b0 in range(0, len(idx), batch):
            bi = idx[b0:b0 + batch]
            x = flow.inner.embed_tokens(torch.from_numpy(ids[bi]).to("mps"))
            pid = torch.arange(x.shape[1], device=x.device)[None]
            ctx = flow._ctx(x, pid)
            for l in range(last + 1):
                y = flow.layer(l, x, ctx)
                if l in group:
                    xs = x[:, C.FIRST_POS:].float().reshape(-1, d)
                    eps = flow.layers[l].input_layernorm.variance_epsilon
                    z = xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)
                    z = torch.cat([z, torch.ones_like(z[:, :1])], 1)
                    u = (y[:, C.FIRST_POS:].float() - x[:, C.FIRST_POS:].float()).reshape(-1, d)
                    a = acc[(l, s)]
                    a["Szz"] += (z.T @ z).cpu().double().numpy()
                    a["Szu"] += (z.T @ u).cpu().double().numpy()
                    a["Suu"] += (u.T @ u).cpu().double().numpy()
                    a["n"] += z.shape[0]
                x = y
            ntok += len(bi) * ids.shape[1]
        log(f"  layers {group[0]}-{last}: {s} done, {time.time() - t0:.0f}s")
    return acc, ntok / (time.time() - t0)


@torch.no_grad()
def hidden_check(flow, ids):
    """Max |difference| between this script's layer inputs (flow.layer, one at a time) and the
    library's output_hidden_states on one chunk, layers 0..n-1 (the last entry of the library's
    list is after the final norm and is skipped)."""
    x0 = torch.from_numpy(ids[:1]).to("mps")
    hs = flow.model(x0, output_hidden_states=True).hidden_states
    x = flow.inner.embed_tokens(x0)
    ctx = flow._ctx(x, torch.arange(x.shape[1], device=x.device)[None])
    err = []
    for l in range(flow.n_layers):
        err.append(float((x.float() - hs[l].float()).abs().max() /
                         hs[l].float().abs().max()))
        x = flow.layer(l, x, ctx)
    return max(err)


def checks(acc, d):
    """Exact identities of the sums: the constant coordinate's sums equal n, and the mean of
    z_i^2 over the d coordinates is mean(x^2)/(mean(x^2)+eps), just below 1."""
    out = {}
    for (l, s), a in acc.items():
        n = float(a["n"])
        out[f"L{l:02d}_{s}"] = {
            "n": n,
            "const_sum_minus_n": float(a["Szz"][-1, -1] - n),
            "mean_z2": float(np.trace(a["Szz"][:d, :d]) / (n * d)),
            "mean_u2": float(np.trace(a["Suu"]) / (n * d)),
            "u_mean_norm": float(np.linalg.norm(a["Szu"][-1] / n)),
            "symmetric_err": float(np.abs(a["Szz"] - a["Szz"].T).max()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(C.MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=8, help="chunks per forward pass")
    p.add_argument("--max_chunks", type=int, default=None, help="smoke runs only")
    p.add_argument("--smoke", action="store_true", help="do not write the cache or results")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = C.out_dir(args.model, "task1")
    if args.smoke:
        out = os.path.join(out, "smoke")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")

    def log(m):
        print(f"{time.time() - t0:.0f}s {m}", flush=True)

    m, tok = FL.load(C.MODELS[args.model])
    flow = FL.Flow(m)
    d, n_layers = flow.cfg.hidden_size, flow.n_layers
    ids, info = C.chunks(tok, args.seed, log)
    if args.max_chunks:  # smoke run: the first max_chunks/3 chunks of each split
        C.SPLITS = {s: list(r)[:max(1, args.max_chunks // 3)] for s, r in C.SPLITS.items()}
    hcheck = hidden_check(flow, ids)
    log(f"layer inputs vs library hidden states: max relative error {hcheck:.2e}")
    per_layer = 8 * ((d + 1) ** 2 + (d + 1) * d + d * d) * len(C.SPLITS)
    gsize = max(1, min(n_layers, int(MEM_BUDGET // per_layer)))
    groups = [list(range(g, min(g + gsize, n_layers))) for g in range(0, n_layers, gsize)]
    log(f"{C.MODELS[args.model]} d={d} layers={n_layers}; {len(groups)} pass(es) of "
        f"{gsize} layers; accumulators {per_layer * gsize / 1e9:.1f} GB per pass")
    cdir = os.path.join(C.CACHE, args.model)
    os.makedirs(cdir, exist_ok=True)
    allchk, speeds = {}, []
    for g in groups:
        acc, tps = run(flow, ids, g, args.batch, log)
        speeds.append(tps)
        allchk.update(checks(acc, d))
        if not args.smoke:
            for (l, s), a in acc.items():
                np.savez(C.acc_path(args.model, l, s), **a)
        log(f"pass layers {g[0]}-{g[-1]}: {tps:.0f} tokens/s (forward to layer {g[-1]})")
        del acc
    worst = {
        "const_sum_minus_n": max(abs(v["const_sum_minus_n"]) for v in allchk.values()),
        "mean_z2_min": min(v["mean_z2"] for v in allchk.values()),
        "mean_z2_max": max(v["mean_z2"] for v in allchk.values()),
        "symmetric_err": max(v["symmetric_err"] for v in allchk.values()),
    }
    res = {"model": C.MODELS[args.model], "seed": args.seed, "d": d, "n_layers": n_layers,
           "data": {"dataset": "Salesforce/wikitext wikitext-103-raw-v1 train",
                    "n_tokens": info["n_tokens"], "n_chunks_available": info["n_chunks_available"],
                    "chunk_len": C.CHUNK_LEN, "positions": [C.FIRST_POS, C.CHUNK_LEN - 1],
                    "splits": {s: [min(r), max(r)] for s, r in C.SPLITS.items()},
                    "positions_per_split": {s: allchk[f"L00_{s}"]["n"] for s in C.SPLITS}},
           "chunk_offsets": info["chunk_offsets"],
           "passes": [[g[0], g[-1]] for g in groups], "tokens_per_second": speeds,
           "batch_chunks": args.batch, "checks_worst": worst,
           "hidden_check_max_rel_err": hcheck, "per_layer": allchk,
           "cache": None if args.smoke else os.path.relpath(cdir, C.ROOT),
           "wall_seconds": time.time() - t0}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"done; checks {worst}")


if __name__ == "__main__":
    main()
