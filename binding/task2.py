"""Binding task 2: top-k transcoders for every MLP, and their fidelity (binding/PLAN.md).

Training (recipe T1: 16,384 units, k = 32, 20M positions; T2: 32,768, k = 64, 40M):
- input h = ln2(x) (the MLP's own LayerNorm output), target y = mlp(h), at every position;
- the six layers' transcoders train together from one stream of forward passes;
- stream: each step is 16 sequences of 256 tokens (4,096 positions), drawn fresh from a generator
  seeded with the transcoder seed: 8 random windows of the STORY train split (the Phase 2 STORY
  rows are from the validation split, so never in the stream) and 8 rows of packed biographies
  (<|endoftext|> + biography, concatenated and cut to 256) of the run's trained people, rendered
  with the run's training templates (template 5 for even-index people when the run used it).
  main-0 and the untrained model: 16 STORY windows. Activations are never stored;
- decoder rows unit norm (renormalised after every step); W_enc initialised to W_dec^T; b_dec
  initialised to the mean of y, and the variance of y in the loss fixed, both from 8 batches
  drawn with seed + 1000 before training;
- loss = mean squared error / variance of y; Adam lr 5e-4, linear warmup over the first 1% of
  steps, then constant.

Fidelity (--eval), on held-out data:
- FVU per layer on 1,000 held-out STORY rows (validation split) and the FACT prompts of people
  0..999 (all positions of prompt + answer);
- dead units: never active on the 1,000,192 held-out STORY positions nor on the FACT prompts;
- FACT recall (Phase 2 FACT set) and STORY loss (Phase 2 STORY rows) with one layer's MLP replaced
  by its transcoder at a time, and with all six replaced; unedited values alongside;
- gate F1 per layer: FVU <= 0.15, recall >= 0.85 x unedited, STORY loss up <= 0.10, dead <= 30%.

  python binding/task2.py --run P2-main-2k-s0 --tc 0 --recipe T1 --max_steps 50   (smoke)
  python binding/task2.py --run P2-main-2k-s0 --tc 0 --recipe T1
  python binding/task2.py --run P2-main-2k-s0 --tc 0 --eval
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

SEQ, N_SEQ = 256, 16
LR, WARM = 5e-4, 0.01


class Stream:
    def __init__(self, run, seed):
        self.rng = np.random.default_rng(seed)
        self.story = np.memmap(os.path.join(B.ROOT, "data", "cache", "tokens", "stories_train.bin"),
                               dtype=np.uint16, mode="r")
        self.tp = B.trained_people(run)
        self.t5 = bool(B.meta(run).get("template5_even"))
        self.buf = []

    def story_rows(self, n):
        s = self.rng.integers(0, len(self.story) - SEQ, n)
        return np.stack([np.asarray(self.story[i:i + SEQ], dtype=np.int64) for i in s])

    def bio_rows(self, n):
        t = B.tok_ids()
        rows = []
        while len(rows) < n:
            while len(self.buf) < SEQ:
                i = int(self.rng.integers(len(self.tp)))
                nt = B.n_templates_for(i) if self.t5 else B.N_TRAIN_TEMPLATES
                self.buf += [t["eot"]] + t["tok"].encode(
                    B.render_bio(self.tp[i], self.rng, n_templates=nt)).ids
            rows.append(self.buf[:SEQ])
            self.buf = self.buf[SEQ:]
        return np.array(rows, dtype=np.int64)

    def batch(self):
        if self.tp:
            x = np.concatenate([self.story_rows(N_SEQ // 2), self.bio_rows(N_SEQ // 2)])
        else:
            x = self.story_rows(N_SEQ)
        return torch.from_numpy(x).to(B.DEVICE)


class Capture:
    """Forward hooks on every blk.mlp: (h, y) per layer."""

    def __init__(self, model):
        self.model, self.hs, self.ys, self.h = model, {}, {}, []

    def __enter__(self):
        for l, blk in enumerate(self.model.blocks):
            def hook(mod, args, out, l=l):
                self.hs[l] = args[0].detach()
                self.ys[l] = out.detach()
            self.h.append(blk.mlp.register_forward_hook(hook))
        return self

    def __exit__(self, *e):
        for h in self.h:
            h.remove()


def train(args):
    rec = B.RECIPES[args.recipe]
    torch.manual_seed(args.tc)
    model = B.load(args.run)
    for p in model.parameters():
        p.requires_grad_(False)
    n_steps = rec["positions"] // (N_SEQ * SEQ)
    if args.max_steps:
        n_steps = min(n_steps, args.max_steps)
    tcs = [B.Transcoder(rec["n_units"], rec["k"]).to(B.DEVICE) for _ in range(B.N_LAYER)]
    # b_dec and loss variance from 8 separate batches
    init = Stream(args.run, args.tc + 1000)
    ys_all = [[] for _ in tcs]
    with torch.no_grad(), Capture(model) as cap:
        for _ in range(8):
            model(init.batch())
            for l in range(B.N_LAYER):
                ys_all[l].append(cap.ys[l].reshape(-1, B.D_MODEL).cpu().double())
    var = []
    for l, tc in enumerate(tcs):
        y = torch.cat(ys_all[l])
        mu = y.mean(0)
        tc.b_dec.data.copy_(mu.float())
        var.append(float(((y - mu) ** 2).sum(1).mean()))
    del ys_all
    opt = torch.optim.Adam([p for tc in tcs for p in tc.parameters()], lr=LR)
    warm = max(1, int(WARM * n_steps))
    stream = Stream(args.run, args.tc)
    t0 = time.time()
    log = []
    for step in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / warm)
        with torch.no_grad(), Capture(model) as cap:
            model(stream.batch())
        loss_l = []
        tot = 0
        for l, tc in enumerate(tcs):
            h = cap.hs[l].reshape(-1, B.D_MODEL)
            y = cap.ys[l].reshape(-1, B.D_MODEL)
            loss = ((tc(h) - y) ** 2).sum(1).mean() / var[l]
            tot = tot + loss
            loss_l.append(loss.item() if step % 100 == 0 or step == n_steps - 1 else None)
        opt.zero_grad(set_to_none=True)
        tot.backward()
        opt.step()
        for tc in tcs:
            tc.renorm()
        if step % 100 == 0 or step == n_steps - 1:
            el = time.time() - t0
            log.append({"step": step, "loss": loss_l, "seconds": el,
                        "positions_per_s": (step + 1) * N_SEQ * SEQ / el})
            print(f"{el:.0f}s step {step}/{n_steps} nmse " +
                  " ".join(f"{x:.3f}" for x in loss_l) +
                  f"  {(step + 1) * N_SEQ * SEQ / el:.0f} pos/s", flush=True)
    d = os.path.join(B.CKPT, args.run, f"tc{args.tc}" + ("_smoke" if args.max_steps else ""))
    os.makedirs(d, exist_ok=True)
    for l, tc in enumerate(tcs):
        torch.save(tc.state_dict(), os.path.join(d, f"L{l}.pt"))
    with open(os.path.join(d, "recipe.json"), "w") as f:
        json.dump({**rec, "name": args.recipe, "steps": n_steps, "seed": args.tc,
                   "y_variance": var}, f, indent=1)
    out = B.out_dir("task2", args.run)
    name = f"train_tc{args.tc}" + ("_smoke" if args.max_steps else "")
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump({"run": args.run, "tc": args.tc, "recipe": args.recipe, "steps": n_steps,
                   "log": log, "wall_seconds": time.time() - t0}, f, indent=1)


def fact_rows(meta):
    """FACT prompts (prompt + answer) as padded rows with a valid-position mask."""
    pr = ac.fact_prompts(meta)
    L = max(len(p) + len(a) for _, _, p, a, _ in pr)
    x = np.zeros((len(pr), L), dtype=np.int64)
    m = np.zeros((len(pr), L), dtype=bool)
    for r, (_, _, p, a, _) in enumerate(pr):
        s = p + a
        x[r, :len(s)] = s
        m[r, :len(s)] = True
    return x, m


@torch.no_grad()
def evaluate(args):
    model = B.load(args.run)
    meta = B.meta(args.run)
    tcs, rec = B.load_transcoders(args.run, args.tc)
    t0 = time.time()
    n_units = rec["n_units"]
    # FVU on 1,000 held-out STORY rows + FACT prompts
    story = B.heldout_story_rows()
    fx, fm = fact_rows(meta)
    err = np.zeros(B.N_LAYER)
    ysum = np.zeros((B.N_LAYER, B.D_MODEL))
    ysq = np.zeros(B.N_LAYER)
    n = 0
    fired = torch.zeros(B.N_LAYER, n_units, device=B.DEVICE)
    fired_story = torch.zeros(B.N_LAYER, n_units, device=B.DEVICE)

    def run_rows(x, mask, fvu, count_into):
        nonlocal n
        with Capture(model) as cap:
            model(torch.from_numpy(x).to(B.DEVICE))
        mk = torch.from_numpy(mask).to(B.DEVICE)
        for l, tc in enumerate(tcs):
            h, y = cap.hs[l][mk], cap.ys[l][mk]
            v, i = tc.encode(h)
            act = torch.zeros(n_units, device=B.DEVICE)
            act.index_add_(0, i[v > 0], torch.ones_like(v[v > 0]))
            for c in count_into:
                c[l] += act
            if fvu:
                yh = tc.decode(v, i)
                yc = y.cpu().double()
                err[l] += float(((yh.cpu().double() - yc) ** 2).sum())
                ysum[l] += yc.sum(0).numpy()
                ysq[l] += float((yc ** 2).sum())
        if fvu:
            n += int(mask.sum())

    for i in range(0, len(story), 64):
        x = story[i:i + 64]
        run_rows(x, np.ones_like(x, dtype=bool), i < 1000, [fired, fired_story])
    for i in range(0, len(fx), 512):
        run_rows(fx[i:i + 512], fm[i:i + 512], True, [fired])
    fvu = [float(err[l] / (ysq[l] - (ysum[l] ** 2).sum() / n)) for l in range(B.N_LAYER)]
    dead = (fired == 0).float().mean(1).cpu().numpy().tolist()
    dead_story = (fired_story == 0).float().mean(1).cpu().numpy().tolist()
    # replacement
    base = {"recall": ac.fact_recall(model, meta)[0], "story_loss": ac.mean_loss(model, "STORY", meta)}
    rep = {}
    for name, layers in [(f"L{l}", [l]) for l in range(B.N_LAYER)] + [("all", list(range(6)))]:
        with B.Replace(model, tcs, layers):
            rep[name] = {"recall": ac.fact_recall(model, meta)[0],
                         "story_loss": ac.mean_loss(model, "STORY", meta)}
    gate = {}
    for l in range(B.N_LAYER):
        r = rep[f"L{l}"]
        checks = {"fvu": fvu[l] <= 0.15,
                  "recall": r["recall"] >= 0.85 * base["recall"],
                  "story_loss": r["story_loss"] - base["story_loss"] <= 0.10,
                  "dead": dead[l] <= 0.30}
        gate[f"L{l}"] = {"pass": all(checks.values()), **checks}
    res = {"run": args.run, "tc": args.tc, "recipe": rec, "n_fvu_positions": n,
           "fvu": fvu, "dead_frac": dead, "dead_frac_story_only": dead_story,
           "unedited": base, "replaced": rep, "gate_F1": gate,
           "gate_F1_all_layers_pass": all(g["pass"] for g in gate.values()),
           "recall_meaningful": bool(meta.get("n_people")),
           "wall_seconds": time.time() - t0}
    out = B.out_dir("task2", args.run)
    with open(os.path.join(out, f"metrics_tc{args.tc}.json" if args.tc else "metrics.json"),
              "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("fvu", "dead_frac", "unedited", "gate_F1_all_layers_pass")}),
          flush=True)
    for l in range(B.N_LAYER):
        print(f"  L{l} recall {rep[f'L{l}']['recall']:.3f} story {rep[f'L{l}']['story_loss']:.4f}"
              f" {gate[f'L{l}']}")
    print(f"  all six: {rep['all']}")


def table(runs):
    rows = []
    for run, tc in runs:
        p = os.path.join(B.RES, "task2", run, f"metrics_tc{tc}.json" if tc else "metrics.json")
        with open(p) as f:
            m = json.load(f)
        for l in range(B.N_LAYER):
            r = m["replaced"][f"L{l}"]
            rows.append({"run": run, "tc": tc, "recipe": m["recipe"]["name"], "layer": l,
                         "fvu": m["fvu"][l], "dead_frac": m["dead_frac"][l],
                         "recall_unedited": m["unedited"]["recall"],
                         "recall_replaced": r["recall"],
                         "story_loss_unedited": m["unedited"]["story_loss"],
                         "story_loss_replaced": r["story_loss"],
                         "recall_all_six": m["replaced"]["all"]["recall"],
                         "story_loss_all_six": m["replaced"]["all"]["story_loss"],
                         "F1_pass": m["gate_F1"][f"L{l}"]["pass"],
                         "recall_meaningful": m["recall_meaningful"]})
    with open(os.path.join(B.out_dir("task2"), "fidelity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run")
    p.add_argument("--tc", type=int, default=0)
    p.add_argument("--recipe", default="T1", choices=list(B.RECIPES))
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="unused; the transcoder seed is --tc")
    p.add_argument("--table", nargs="*", help="run:tc pairs for fidelity.csv")
    args = p.parse_args()
    if args.table is not None:
        table([(x.split(":")[0], int(x.split(":")[1])) for x in args.table])
        return
    out = B.out_dir("task2", args.run)
    name = ("cmd_eval" if args.eval else "cmd_train") + f"_tc{args.tc}.txt"
    with open(os.path.join(out, name), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    evaluate(args) if args.eval else train(args)


if __name__ == "__main__":
    main()
