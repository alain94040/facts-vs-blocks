"""Quick probe (circuit/PROBE.md): does the model answer a small program as its single next token?

Usage: python circuit/probe/probe.py --model Qwen/Qwen2.5-7B-Instruct --seed 0
Writes results/circuit/probe/<model short name>/{metrics.json, cmd.txt, differential.csv}.

Plain text, no chat template. Prompt = program + "# output: "; the answer is the next token.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

OPS = ["+", "-", "*"]


def apply(x, op, y):
    return x + y if op == "+" else x - y if op == "-" else x * y


def program(a0=4, l2=3, l3=2, op2="+", op3="*", op4="-", overwrite=True):
    """The PROBE.md program with every literal and operator as a parameter. Returns (text, answer, values)."""
    a = a0
    b = apply(a, op2, l2)
    v3 = apply(b, op3, l3)
    c = apply(v3, op4, b)
    v = "a" if overwrite else "d"
    text = (f"a = {a0}\nb = a {op2} {l2}\n{v} = b {op3} {l3}\nc = {v} {op4} b\nprint(c)\n")
    return text, c, (a, b, v3, c)


# Three worked examples of the same shape, different values (answers 8, 3, 5; none is 7).
SHOTS = [program(3, 1, 3), program(2, 1, 2), program(1, 4, 2)]
SUFFIX = "# output: "


def fewshot_prefix():
    return "".join(f"{t}{SUFFIX}{ans}\n\n" for t, ans, _ in SHOTS)


def random_programs(n, seed):
    """Same shape: a = L1; b = a op L2; a = b op L3 (overwrite); c = a op b (reads it); print(c).
    Literals 0..9, intermediates 0..20, answer 0..9 and not equal to any literal."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        l1, l2, l3 = (rng.randint(0, 9) for _ in range(3))
        op2, op3, op4 = (rng.choice(OPS) for _ in range(3))
        text, c, vals = program(l1, l2, l3, op2, op3, op4)
        if not all(0 <= v <= 20 for v in vals) or not 0 <= c <= 9 or c in (l1, l2, l3):
            continue
        out.append({"program": text, "answer": c})
    return out


class Model:
    def __init__(self, name, dtype):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("mps").eval()
        self.seconds = []

    @torch.no_grad()
    def top5(self, prompt):
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to("mps")
        t0 = time.time()
        logits = self.model(ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        v, i = p.topk(5)
        v, i = v.cpu().tolist(), i.cpu().tolist()
        self.seconds.append(time.time() - t0)
        return [(self.tok.decode([t]), pr) for t, pr in zip(i, v)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    short = args.model.split("/")[-1]
    out_dir = os.path.join(ROOT, "results", "circuit", "probe", short)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")

    m = Model(args.model, getattr(torch, args.dtype))
    m.top5("a = 1\n")  # warm-up, not timed in the report
    m.seconds.clear()
    prefix = fewshot_prefix()
    metrics = {"model": args.model, "dtype": args.dtype, "seed": args.seed,
               "fewshot_prefix": prefix}

    # Step 1: the exact program.
    text, ans, _ = program()
    assert ans == 7
    metrics["exact"] = {"answer": ans,
                        "zero_shot_top5": m.top5(text + SUFFIX),
                        "three_shot_top5": m.top5(prefix + text + SUFFIX)}

    # Step 2: differential table, 3-shot.
    variants = [(f"(a) 4->{k}", dict(a0=k)) for k in range(1, 10) if k != 4]
    variants += [(f"(b) 3->{k}", dict(l2=k)) for k in range(1, 10) if k != 3]
    variants += [(f"(c) 2->{k}", dict(l3=k)) for k in range(1, 5) if k != 2]
    for slot, orig in (("op2", "+"), ("op3", "*"), ("op4", "-")):
        variants += [(f"(d) line{slot[-1]} {orig}->{o}", {slot: o}) for o in OPS if o != orig]
    variants += [("(e) no overwrite (d = b * 2; c = d - b)", dict(overwrite=False))]
    rows, skipped = [], []
    for name, kw in variants:
        text, ans, _ = program(**kw)
        if not 0 <= ans <= 9:
            skipped.append({"variant": name, "answer": ans})
            continue
        top = m.top5(prefix + text + SUFFIX)
        tok, pr = top[0]
        rows.append({"variant": name, "true_answer": ans, "top_token": tok.strip(),
                     "probability": round(pr, 4), "correct": tok.strip() == str(ans),
                     "p_true": round(next((p for t, p in top if t.strip() == str(ans)), 0.0), 4)})
    with open(os.path.join(out_dir, "differential.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    metrics["differential"] = {"n_run": len(rows), "n_correct": sum(r["correct"] for r in rows),
                               "n_skipped": len(skipped), "skipped": skipped}

    # Steps 3-4: 50 random programs and shortcut baselines.
    progs = random_programs(args.n, args.seed)
    res, last_ok, freq_ok = [], 0, 0
    for p in progs:
        prompt = prefix + p["program"] + SUFFIX
        top = m.top5(prompt)
        digits = [ch for ch in prompt if ch.isdigit()]
        last = int(digits[-1])
        freq = int(Counter(digits).most_common(1)[0][0])  # ties: first seen
        last_ok += last == p["answer"]
        freq_ok += freq == p["answer"]
        res.append({**p, "top5": top, "correct": top[0][0].strip() == str(p["answer"]),
                    "baseline_last": last, "baseline_freq": freq})
    metrics["random"] = {"n": len(res), "accuracy": sum(r["correct"] for r in res) / len(res),
                         "baseline_last_digit": last_ok / len(res),
                         "baseline_most_frequent_digit": freq_ok / len(res),
                         "answer_counts": dict(sorted(Counter(p["answer"] for p in progs).items())),
                         "items": res}
    metrics["seconds_per_prompt"] = sum(m.seconds) / len(m.seconds)
    metrics["n_prompts_timed"] = len(m.seconds)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps({k: metrics[k] for k in ("exact", "differential", "seconds_per_prompt")}, indent=1))
    print({k: v for k, v in metrics["random"].items() if k != "items"})


if __name__ == "__main__":
    main()
