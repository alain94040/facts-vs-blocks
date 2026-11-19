"""Probe 2 (circuit/PROBE2.md): which part of the program breaks the single-step answer?

Usage: python circuit/probe/probe2.py --model Qwen/Qwen2.5-7B-Instruct --seed 0
Writes results/circuit/probe2/<model short name>/{metrics.json, cmd.txt, levels.csv}.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe  # noqa: E402  (sets HF_HOME and the MPS fallback)
from probe import ROOT, SUFFIX, Model, apply  # noqa: E402

import torch  # noqa: E402

NAMES = list("abcdexyz")
LEVELS = ["C1", "C2", "C3", "A1", "A2", "A3", "A3x", "A4"]


def make_item(level, rng, idx):
    """One item: dict(program, answer, literals) or None if it breaks the value constraints."""
    names = rng.sample(NAMES, len(NAMES))  # without replacement
    u, v, w = names[:3]
    L = [rng.randint(0, 9) for _ in range(3)]
    if level in ("C1", "C2", "C3"):
        if level == "C1":
            lines = [f"{u} = {L[0]}", f"{v} = {L[1]}", f"{u} = {v}", f"{v} = {L[2]}"]
            out, ans = u, L[1]
            lits = L
        else:
            lines = [f"{u} = {L[0]}", f"{v} = {L[1]}", f"{w} = {u}", f"{u} = {v}", f"{v} = {L[2]}"]
            out, ans = (w, L[0]) if idx % 2 == 0 else (u, L[1])  # half each
            lits = list(L)
            if level == "C3":  # two distractor lines, new variables and literals, random positions
                for name in names[3:5]:
                    lit = rng.randint(0, 9)
                    lits.append(lit)
                    lines.insert(rng.randint(0, len(lines)), f"{name} = {lit}")
                # literals in program order, for the last-literal baseline
                lits = [int(x.split(" = ")[1]) for x in lines if x.split(" = ")[1].isdigit()]
        text = "\n".join(lines) + f"\nprint({out})\n"
        return {"program": text, "answer": ans, "literals": lits}

    ops = ["+", "-", "*"] if level in ("A3x", "A4") else ["+", "-"]
    o = [rng.choice(ops) for _ in range(3)]
    if level == "A1":
        vals = [L[0], apply(L[0], o[0], L[1])]
        lines, out, lits = [f"{u} = {L[0]}", f"{v} = {u} {o[0]} {L[1]}"], v, L[:2]
    elif level == "A2":
        x = apply(L[0], o[0], L[1])
        vals = [L[0], x, apply(x, o[1], L[2])]
        lines, out, lits = [f"{u} = {L[0]}", f"{v} = {u} {o[0]} {L[1]}", f"{w} = {v} {o[1]} {L[2]}"], w, L
    elif level in ("A3", "A3x"):
        x = apply(L[0], o[0], L[1])
        vals = [L[0], x, apply(x, o[1], L[2])]
        lines, out, lits = [f"{u} = {L[0]}", f"{v} = {u} {o[0]} {L[1]}", f"{u} = {v} {o[1]} {L[2]}"], u, L
    else:  # A4: Probe 1 shape with drawn names
        x = apply(L[0], o[0], L[1])
        y = apply(x, o[1], L[2])
        vals = [L[0], x, y, apply(y, o[2], x)]
        lines = [f"{u} = {L[0]}", f"{v} = {u} {o[0]} {L[1]}", f"{u} = {v} {o[1]} {L[2]}",
                 f"{w} = {u} {o[2]} {v}"]
        out, lits = w, L
    ans = vals[-1]
    if not all(0 <= t <= 20 for t in vals) or not 0 <= ans <= 9 or ans in lits:
        return None
    return {"program": "\n".join(lines) + f"\nprint({out})\n", "answer": ans, "literals": lits}


def draw(level, n, seed, exclude=()):
    rng = random.Random(seed)
    items, i = [], 0
    while len(items) < n:
        it = make_item(level, rng, i)
        if it is not None and it["program"] not in exclude:
            items.append(it)
            i += 1
    return items


def baselines(items):
    last = freq = rnd = 0.0
    for it in items:
        lits = it["literals"]
        digits = [ch for ch in it["program"] if ch.isdigit()]
        last += lits[-1] == it["answer"]
        freq += int(Counter(digits).most_common(1)[0][0]) == it["answer"]  # ties: first seen
        rnd += 1 / len(set(lits))
    n = len(items)
    return {"last_literal": last / n, "most_frequent_digit": freq / n, "random_literal": rnd / n}


@torch.no_grad()
def score(m, prompt, answer, digit_ids):
    ids = m.tok(prompt, return_tensors="pt")["input_ids"].to("mps")
    t0 = time.time()
    p = torch.softmax(m.model(ids).logits[0, -1].float(), -1)
    v, i = p.topk(5)
    top = [(m.tok.decode([t]), pr) for t, pr in zip(i.cpu().tolist(), v.cpu().tolist())]
    p_true = p[digit_ids[answer]].item()
    m.seconds.append(time.time() - t0)
    return top, p_true


def first_digit_after_output(gen):
    """First digit after the first '# output:' in the generated text, or None."""
    k = gen.find("# output:")
    if k < 0:
        return None
    mt = re.search(r"\d", gen[k + len("# output:"):])
    return int(mt.group(0)) if mt else None


def written_steps(m, n, seed):
    """Probe 1 programs; worked examples carry a '# a=3, b=4, a=12, c=8' line; greedy up to 60 tokens."""
    def trace(a0, l2, l3):
        a = a0
        b = a + l2
        a2 = b * l3
        return f"# a={a}, b={b}, a={a2}, c={a2 - b}\n"
    shot_args = [(3, 1, 3), (2, 1, 2), (1, 4, 2)]  # the Probe 1 worked examples
    prefix = ""
    for args in shot_args:
        text, ans, _ = probe.program(*args)
        prefix += f"{text}{trace(*args)}{SUFFIX}{ans}\n\n"
    items, correct, no_line, secs = [], 0, 0, []
    for p in probe.random_programs(n, seed):
        ids = m.tok(prefix + p["program"], return_tensors="pt")["input_ids"].to("mps")
        t0 = time.time()
        with torch.no_grad():
            out = m.model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                   max_new_tokens=60, pad_token_id=m.tok.eos_token_id)
        secs.append(time.time() - t0)
        gen = m.tok.decode(out[0, ids.shape[1]:])
        pred = first_digit_after_output(gen)
        if "# output:" not in gen:
            no_line += 1
        correct += pred == p["answer"]
        items.append({**p, "generated": gen, "pred": pred, "correct": pred == p["answer"]})
    return {"n": n, "accuracy": correct / n, "n_no_output_line": no_line,
            "seconds_per_item": sum(secs) / len(secs), "fewshot_prefix": prefix, "items": items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    short = args.model.split("/")[-1]
    out_dir = os.path.join(ROOT, "results", "circuit", "probe2", short)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    m = Model(args.model, getattr(torch, args.dtype))
    digit_ids = [m.tok.convert_tokens_to_ids(str(d)) for d in range(10)]
    assert all(m.tok.decode([t]) == str(d) for d, t in enumerate(digit_ids))
    m.top5("a = 1\n")  # warm-up
    m.seconds.clear()

    metrics = {"model": args.model, "dtype": args.dtype, "seed": args.seed, "levels": {}}
    rows = []
    for level in LEVELS:
        tests = draw(level, args.n, args.seed)
        shots = draw(level, 3, 1, exclude={t["program"] for t in tests})
        prefix = "".join(f"{s['program']}{SUFFIX}{s['answer']}\n\n" for s in shots)
        res = []
        for it in tests:
            top, p_true = score(m, prefix + it["program"] + SUFFIX, it["answer"], digit_ids)
            res.append({**it, "top5": top, "p_true": p_true,
                        "correct": top[0][0].strip() == str(it["answer"])})
        tops = Counter(r["top5"][0][0] for r in res).most_common(3)
        row = {"level": level, "n": len(res), "accuracy": sum(r["correct"] for r in res) / len(res),
               "mean_p_true": sum(r["p_true"] for r in res) / len(res),
               "top_tokens": " ".join(f"{t!r}:{c}" for t, c in tops)}
        if level.startswith("C"):
            row.update(baselines(tests))
        rows.append(row)
        metrics["levels"][level] = {**row, "fewshot_prefix": prefix,
                                    "answer_counts": dict(sorted(Counter(t["answer"] for t in tests).items())),
                                    "items": res}
        print(row, flush=True)
    metrics["seconds_per_prompt"] = sum(m.seconds) / len(m.seconds)

    metrics["written_steps"] = written_steps(m, 50, args.seed)
    ws = metrics["written_steps"]
    print({k: ws[k] for k in ("accuracy", "n_no_output_line", "seconds_per_item")}, flush=True)

    fields = ["level", "n", "accuracy", "last_literal", "most_frequent_digit", "random_literal",
              "mean_p_true", "top_tokens"]
    with open(os.path.join(out_dir, "levels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
