"""Circuit task 1: screening (circuit/PLAN.md). Plain transformers forward passes, no transcoders.

Usage: python circuit/screen.py --model Qwen/Qwen3-4B --seed 0 [--limit 50]
       python circuit/screen.py --summarise --seed 0
Per model: results/circuit/task1/<model short>/{items.csv, metrics.json, cmd.txt}.
Summary:   results/circuit/task1/{accuracy.csv, baselines.json, metrics.json, accuracy.png}.

300 items per family per level, seeds 1-300 (SCREEN). Accuracy = the gold token is the top next
token. Mean log-prob of the gold token. Batched with right padding; the score is read at each
prompt's last real token. Default batch 1: in bfloat16 batching changed gold log-probs by up to
0.37 and flipped 1 of 100 top tokens in a smoke test, so items run one at a time.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tasks  # noqa: E402

OUT = os.path.join(ROOT, "results", "circuit", "task1")
SEEDS = range(1, 301)
MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-7B-Instruct"]
MAIN = "Qwen/Qwen3-4B"
FAMILIES = ["PROG", "STORY", "ARITH", "ORDER", "TWOHOP"]


def all_items(limit=None):
    cache = os.path.join(ROOT, "data", "cache", "circuit", "screen_items.json")
    if os.path.exists(cache) and limit is None:
        return json.load(open(cache))
    seeds = list(SEEDS)[:limit] if limit else SEEDS
    items = [{k: v for k, v in tasks.make(f, lv, s).items() if k in ("family", "level", "seed", "prompt", "answer")}
             for f in FAMILIES for lv in tasks.LEVELS[f] for s in seeds]
    if limit is None:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(items, open(cache, "w"))
    return items


def run_model(name, seed, limit, batch):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(seed)
    short = name.split("/")[-1]
    d = os.path.join(OUT, short if limit is None else short + "_smoke")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
    items = all_items(limit)
    gold = {}
    for it in items:
        ids = tok(it["answer"], add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (name, it["answer"], ids)
        gold[it["answer"]] = ids[0]
    enc = [tok(it["prompt"])["input_ids"] for it in items]
    order = sorted(range(len(items)), key=lambda i: len(enc[i]))
    rows, ntok, t0 = [None] * len(items), 0, time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            L = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), L), tok.pad_token_id or 0, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :len(enc[i])] = torch.tensor(enc[i])
                mask[r, :len(enc[i])] = 1
            logits = model(input_ids=ids.to("mps"), attention_mask=mask.to("mps")).logits
            last = torch.tensor([len(enc[i]) - 1 for i in idx], device="mps")
            lp = torch.log_softmax(logits[torch.arange(len(idx), device="mps"), last].float(), -1).cpu()
            ntok += int(mask.sum())
            for r, i in enumerate(idx):
                g = gold[items[i]["answer"]]
                top = int(lp[r].argmax())
                rows[i] = {"family": items[i]["family"], "level": items[i]["level"],
                           "seed": items[i]["seed"], "answer": items[i]["answer"],
                           "top": tok.decode([top]), "correct": int(top == g),
                           "gold_logprob": round(float(lp[r, g]), 5), "n_tokens": len(enc[i])}
    secs = time.time() - t0
    with open(os.path.join(d, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    acc = {}
    for r in rows:
        a = acc.setdefault(f"{r['family']}/{r['level']}", {"n": 0, "correct": 0, "lp": 0.0})
        a["n"] += 1
        a["correct"] += r["correct"]
        a["lp"] += r["gold_logprob"]
    summary = {k: {"n": v["n"], "accuracy": v["correct"] / v["n"], "mean_logprob": v["lp"] / v["n"]}
               for k, v in acc.items()}
    m = {"model": name, "seed": seed, "dtype": "bfloat16", "batch": batch, "seconds": secs,
         "tokens_per_second": ntok / secs, "prompts_per_second": len(items) / secs,
         "accuracy": summary}
    json.dump(m, open(os.path.join(d, "metrics.json"), "w"), indent=1)
    for k, v in summary.items():
        print(name, k, f"{v['accuracy']:.3f} {v['mean_logprob']:.3f}", flush=True)
    print(f"{secs:.0f}s, {ntok/secs:.0f} tok/s", flush=True)


def chosen_level(accs):
    """accs: list of accuracies in level order. Highest level >= 0.85 whose next lower level is
    also >= 0.85 (the lowest level has no lower level: only its own accuracy counts)."""
    best = None
    for i, a in enumerate(accs):
        if a >= 0.85 and (i == 0 or accs[i - 1] >= 0.85):
            best = i
    return best


def summarise(seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for name in MODELS:
        m = json.load(open(os.path.join(OUT, name.split("/")[-1], "metrics.json")))
        for f in FAMILIES:
            for lv in tasks.LEVELS[f]:
                a = m["accuracy"][f"{f}/{lv}"]
                rows.append({"model": name, "family": f, "level": lv, "n": a["n"],
                             "accuracy": round(a["accuracy"], 4), "mean_logprob": round(a["mean_logprob"], 4)})
    with open(os.path.join(OUT, "accuracy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # shortcut baselines on PROG, no model
    items = [it for it in all_items() if it["family"] == "PROG"]
    base = {}
    for lv in tasks.LEVELS["PROG"]:
        its = [it for it in items if it["level"] == lv]
        last = freq = 0
        for it in its:
            digits = [c for c in it["prompt"] if c.isdigit()]
            last += digits[-1] == it["answer"]
            freq += Counter(digits).most_common(1)[0][0] == it["answer"]  # ties: first seen
        base[lv] = {"n": len(its), "last_digit": last / len(its), "most_frequent_digit": freq / len(its)}
    json.dump(base, open(os.path.join(OUT, "baselines.json"), "w"), indent=1)

    chosen, dropped = {}, []
    for f in FAMILIES:
        accs = [next(r["accuracy"] for r in rows if r["model"] == MAIN and r["family"] == f and r["level"] == lv)
                for lv in tasks.LEVELS[f]]
        i = chosen_level(accs)
        if i is None:
            dropped.append(f)
        else:
            chosen[f] = tasks.LEVELS[f][i]
    prog = chosen.get("PROG")
    g1 = prog is not None and prog != "P1"
    main_task = "PROG" if g1 else next((f for f in ("ORDER", "TWOHOP") if f in chosen), None)
    m = {"seed": seed, "main_model": MAIN, "chosen_levels": chosen, "dropped_families": dropped,
         "gate_G1": {"prog_chosen_level": prog, "pass": g1}, "main_task": main_task,
         "prog_baselines": base}
    json.dump(m, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    with open(os.path.join(OUT, "cmd.txt"), "w") as fh:
        fh.write("python " + " ".join(sys.argv) + "\n")

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(17, 3.6), sharey=True)
    for ax, f in zip(axes, FAMILIES):
        lvs = tasks.LEVELS[f]
        for name, mk in zip(MODELS, ["o-", "s--", "^:"]):
            ys = [next(r["accuracy"] for r in rows if r["model"] == name and r["family"] == f and r["level"] == lv)
                  for lv in lvs]
            ax.plot(lvs, ys, mk, label=name.split("/")[-1])
        if f == "PROG":
            ax.plot(lvs, [base[lv]["last_digit"] for lv in lvs], "x", color="gray", label="last digit")
            ax.plot(lvs, [base[lv]["most_frequent_digit"] for lv in lvs], "+", color="gray",
                    label="most frequent digit")
        ax.axhline(0.85, color="k", lw=0.5, ls="--")
        ax.set_title(f + (f" (chosen {chosen[f]})" if f in chosen else " (dropped)"))
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("accuracy (gold = top token), n=300 per point")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "accuracy.png"), dpi=120)
    print(json.dumps(m, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="seeds per level (smoke test)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        summarise(args.seed)
    else:
        run_model(args.model, args.seed, args.limit, args.batch)


if __name__ == "__main__":
    main()
