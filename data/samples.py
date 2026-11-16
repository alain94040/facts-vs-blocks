"""Gate 1: write 5 random biographies and 5 random stories to results/task1/samples.txt and
measure the average token length of a biography (training templates, 10,000 random people from
set A, excluding the <|endoftext|> separator).

Writes results/task1/samples/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import shlex
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.make_bios import render_bio  # noqa: E402
from data.prepare import load_stories  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_len", type=int, default=10000)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    with open(os.path.join(ROOT, "data", "cache", "bios", "people_A.json")) as f:
        people = json.load(f)
    train = load_stories()["train"]

    bios = [render_bio(people[i], rng) for i in rng.integers(len(people), size=5)]
    story_idx = [int(i) for i in rng.integers(len(train), size=5)]
    stories = [train[i]["text"] for i in story_idx]

    lens = [len(tok.encode(render_bio(people[i], rng)).ids)
            for i in rng.integers(len(people), size=args.n_len)]
    held = [len(tok.encode(render_bio(people[i], rng, heldout=True)).ids)
            for i in rng.integers(len(people), size=args.n_len)]

    lines = ["# Task 1 samples (seed %d)" % args.seed, "", "## 5 random biographies (set A, training templates)", ""]
    for b in bios:
        ids = tok.encode(b).ids
        lines += [b, f"  [{len(ids)} tokens; roundtrip ok: {tok.decode(ids) == b}]", ""]
    lines += ["## 5 random stories (TinyStories train)", ""]
    for i, s in zip(story_idx, stories):
        lines += [f"[train index {i}, {len(tok.encode(s).ids)} tokens]", s.strip(), ""]
    lines += ["## Biography token length", "",
              f"mean over {args.n_len} biographies (training templates): {np.mean(lens):.2f} "
              f"(min {min(lens)}, max {max(lens)})",
              f"mean over {args.n_len} biographies (held-out template 5): {np.mean(held):.2f}", ""]
    out = os.path.join(ROOT, "results", "task1")
    with open(os.path.join(out, "samples.txt"), "w") as f:
        f.write("\n".join(lines))

    res = os.path.join(out, "samples")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    metrics = {"seed": args.seed, "n_bios_measured": args.n_len,
               "bio_mean_tokens_train_templates": float(np.mean(lens)),
               "bio_min_tokens": int(min(lens)), "bio_max_tokens": int(max(lens)),
               "bio_mean_tokens_heldout_template": float(np.mean(held))}
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
