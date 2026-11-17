"""Repeat task 5b: carry over to the 14B model (repeat/PLAN.md). REPORT halves only, no search.

Exactness check (float16, tolerance 5e-2), then the task 5 configurations moved by fraction of
depth: layer i of n_src becomes round(i * n_dst / n_src); a block keeps its start fraction and
its length fraction, rounded to whole layers. Plus the selection-free reference: repeat and
gain(2) on the layer at 50% of depth. Same paired bootstrap against the 14B model's own base.
Skipped (and recorded) if the model does not load or runs under 300 tokens per second.

  python repeat/task5b.py --model confirm --source main --seed 0
"""
import argparse
import csv
import json
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import evaluate as E  # noqa: E402
from repeat import flow as FL  # noqa: E402
from repeat import task5 as T5  # noqa: E402

import torch  # noqa: E402

SPEED_MIN = 300.0


def out_dir(model):
    d = os.path.join(ROOT, "results", "repeat", "task5b")
    return d if model == "confirm" else os.path.join(d, model)


def move(iv, n_src, n_dst):
    iv = dict(iv)
    s, e = iv["start"], iv["end"]
    if e - s == 1:
        ns = round(s * n_dst / n_src)
        iv["start"], iv["end"] = ns, ns + 1
    else:
        ns = round(s * n_dst / n_src)
        nl = max(1, round((e - s) * n_dst / n_src))
        iv["start"], iv["end"] = ns, min(n_dst, ns + nl)
    return iv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(E.MODELS), default="confirm")
    p.add_argument("--source", choices=list(E.MODELS), default="main",
                   help="model whose task 5 configurations are moved")
    p.add_argument("--source_layers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    out = out_dir(args.model)
    os.makedirs(os.path.join(out, "scores"), exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    t0 = time.time()
    res = {"model": E.MODELS[args.model], "source_model": E.MODELS[args.source],
           "seed": args.seed, "split": "REPORT"}
    try:
        m, tok = FL.load(E.MODELS[args.model])
    except Exception as ex:  # noqa: BLE001
        res.update({"skipped": True, "reason": f"model did not load: {ex!r}"})
        json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2)
        print(res)
        return
    flow = FL.Flow(m)
    res["load_seconds"] = time.time() - t0
    res["n_layers"] = flow.n_layers
    res["exactness_float16"] = E.exactness(flow, tok, torch.float16)
    print("exactness", res["exactness_float16"], flush=True)
    with open(os.path.join(ROOT, "results", "repeat", "task0",
                           args.source if args.source != "main" else "main",
                           "metrics.json")) as f:
        n_src = args.source_layers or json.load(f)["n_layers"]
    sets = E.build_sets(tok, E.K_FROZEN)
    t1 = time.time()
    base = E.score(flow, sets, "REPORT")
    dt = time.time() - t1
    res["speed"] = {"tokens": base["_tokens"], "seconds": dt,
                    "tokens_per_second": base["_tokens"] / dt}
    print("speed", res["speed"], flush=True)
    if res["speed"]["tokens_per_second"] < SPEED_MIN:
        res.update({"skipped": True, "reason": f"under {SPEED_MIN} tokens per second"})
        json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=2)
        return
    E.save_scores(os.path.join(out, "scores", "base.npz"), base)
    cfg_src, sel_src = T5.configs(args.source, n_src)
    mid = round(0.5 * flow.n_layers)
    cfg = []
    for target, name, iv in cfg_src:
        if target == "mid":
            continue
        cfg.append((target, name, iv, move(iv, n_src, flow.n_layers)))
    cfg += [("mid", "repeat", None, {"kind": "repeat", "start": mid, "end": mid + 1}),
            ("mid", "gain(2)", None, {"kind": "gain", "g": 2.0, "start": mid, "end": mid + 1})]
    sc, rows = {}, []
    for target, name, src, iv in cfg:
        key = f"{target}:{name}"
        sc[key] = E.score(flow, sets, "REPORT", iv)
        E.save_scores(os.path.join(out, "scores", key.replace(":", "_").replace(" ", "_")
                                   .replace("(", "").replace(")", "") + ".npz"), sc[key])
        rows.append({"target": target, "config": name,
                     "src_start": src["start"] if src else "", "src_end": src["end"] if src else "",
                     "start": iv["start"], "end": iv["end"], "vs": "base",
                     **T5.diff_row(sc[key], base)})
        print(key, iv["start"], iv["end"], round(rows[-1]["d_REASON_logprob"], 3), flush=True)
    for target, a, b in [("layer", "repeat", "gain(2)"), ("block", "repeat", "gain(2)"),
                         ("layer", "attn2", "mlp2"), ("mid", "repeat", "gain(2)")]:
        rows.append({"target": target, "config": f"{a} - {b}", "src_start": "", "src_end": "",
                     "start": "", "end": "", "vs": b,
                     **T5.diff_row(sc[f"{target}:{a}"], sc[f"{target}:{b}"])})
    with open(os.path.join(out, "confirm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    res.update({"skipped": False, "source_layers": n_src, "mid_layer": mid,
                "source_selection": sel_src,
                "moved": {f"{t}:{n}": [iv["start"], iv["end"]] for t, n, _, iv in cfg},
                "base_report": {k: float(v) for k, v in E.summarise(base).items()},
                "wall_seconds": time.time() - t0})
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
