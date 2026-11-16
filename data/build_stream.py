"""Build one training token stream per run (Task 2).

Stories: the first story documents of the TinyStories train split, in their original order, until
300M tokens are reached (identical for every run). Biographies: every selected person appears
exactly --exposures times, each appearance rendered with random training templates (1-4) and
shuffled sentence order, each preceded by <|endoftext|>. Biography documents are inserted at
uniformly random positions between story documents (a uniform random interleaving that keeps the
story order).

Fork runs (--fork_of main-8k): the first 20% of the parent's steps are story tokens only (the
trunk); the remaining stories are interleaved with the biographies. The fork streams of set A
and set B therefore share their first trunk_steps * tokens_per_step tokens exactly.

The stream depends only on the run definition and --seed (default 0), never on the training seed.

Outputs (git-ignored): data/cache/streams/<run>.bin (uint16), data/cache/streams/<run>.json
Writes results/task2/streams/<run>/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.make_bios import render_bio  # noqa: E402

STREAM_DIR = os.path.join(ROOT, "data", "cache", "streams")
TOK_DIR = os.path.join(ROOT, "data", "cache", "tokens")


def tokens_per_step():
    with open(os.path.join(ROOT, "configs", "train.json")) as f:
        c = json.load(f)
    return c["batch_size"] * (c["seq_len"] + 1)  # rows of seq_len+1 tokens, no overlap


def load_people(people_set, n):
    with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{people_set}.json")) as f:
        people = json.load(f)
    assert n <= len(people)
    return people[:n]


def render_bios(people, exposures, rng, tok):
    eot = tok.token_to_id("<|endoftext|>")
    order = rng.permutation(np.repeat(np.arange(len(people)), exposures))
    texts = [render_bio(people[i], rng) for i in order]
    out = []
    for i in range(0, len(texts), 50000):
        out += [np.asarray([eot] + e.ids, np.uint16) for e in tok.encode_batch(texts[i:i + 50000])]
    return out


def interleave(story_docs, bio_docs, rng):
    """Uniform random interleaving that keeps the order within each list."""
    is_bio = np.zeros(len(story_docs) + len(bio_docs), dtype=bool)
    is_bio[rng.choice(len(is_bio), size=len(bio_docs), replace=False)] = True
    out, si, bi = [], 0, 0
    for b in is_bio:
        if b:
            out.append(bio_docs[bi]); bi += 1
        else:
            out.append(story_docs[si]); si += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--people_set", choices=["A", "B"], default="A")
    p.add_argument("--n_people", type=int, default=0)
    p.add_argument("--exposures", type=int, default=100)
    p.add_argument("--story_tokens", type=int, default=300_000_000)
    p.add_argument("--fork_of", default=None, help="parent run whose total steps the fork uses")
    p.add_argument("--trunk_frac", type=float, default=0.2)
    args = p.parse_args()
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    tps = tokens_per_step()

    train = np.memmap(os.path.join(TOK_DIR, "stories_train.bin"), dtype=np.uint16, mode="r")
    offs = np.load(os.path.join(TOK_DIR, "stories_train_offsets.npy"))
    ends = np.append(offs[1:], len(train))
    n_story = int(np.searchsorted(ends, args.story_tokens) + 1)  # docs until >= story_tokens
    story_docs = [train[offs[i]:ends[i]] for i in range(n_story)]
    n_story_tok = int(ends[n_story - 1])

    people = load_people(args.people_set, args.n_people) if args.n_people else []
    bio_docs = render_bios(people, args.exposures, rng, tok) if people else []
    n_bio_tok = int(sum(len(d) for d in bio_docs))

    meta = {"run": args.run, "seed": args.seed, "people_set": args.people_set if people else None,
            "n_people": len(people), "exposures": args.exposures if people else 0,
            "n_story_docs": n_story, "story_tokens": n_story_tok,
            "n_bio_docs": len(bio_docs), "bio_tokens": n_bio_tok,
            "mean_bio_tokens": n_bio_tok / max(1, len(bio_docs)), "tokens_per_step": tps}

    if args.fork_of:
        with open(os.path.join(STREAM_DIR, f"{args.fork_of}.json")) as f:
            parent = json.load(f)
        total_steps = parent["steps"]
        trunk_steps = int(round(args.trunk_frac * total_steps))
        trunk_tok = trunk_steps * tps
        cum = np.cumsum([len(d) for d in story_docs])
        k = int(np.searchsorted(cum, trunk_tok) + 1)  # story docs covering the trunk
        docs = story_docs[:k] + interleave(story_docs[k:], bio_docs, rng)
        n_tok = int(sum(len(d) for d in docs))
        if n_tok < total_steps * tps:
            raise SystemExit(f"fork stream too short: {n_tok} < {total_steps * tps}")
        first_bio_pos = int(cum[k - 1])
        meta.update({"fork_of": args.fork_of, "steps": total_steps, "trunk_steps": trunk_steps,
                     "trunk_tokens": trunk_tok, "first_bio_token": first_bio_pos,
                     "unused_tail_tokens": n_tok - total_steps * tps})
    else:
        docs = interleave(story_docs, bio_docs, rng)
        n_tok = int(sum(len(d) for d in docs))
        meta["steps"] = n_tok // tps

    meta["stream_tokens"] = n_tok
    meta["unused_tail_tokens"] = n_tok - meta["steps"] * tps
    os.makedirs(STREAM_DIR, exist_ok=True)
    path = os.path.join(STREAM_DIR, f"{args.run}.bin")
    with open(path, "wb") as f:
        for i in range(0, len(docs), 100000):
            np.concatenate(docs[i:i + 100000]).astype(np.uint16).tofile(f)
    assert os.path.getsize(path) == 2 * n_tok
    meta["wall_seconds"] = time.time() - t0
    with open(os.path.join(STREAM_DIR, f"{args.run}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    res = os.path.join(ROOT, "results", "task2", "streams", args.run)
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
