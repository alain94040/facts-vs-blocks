"""Task 4b fine-tuning stream (PLAN.md Phase 2b).

Set-B people 0..999 rendered 300 times each with the Task 2b template rules (even index: templates
1-5; odd index: templates 1-4), each preceded by <|endoftext|>, interleaved uniformly at the
document level (data/build_stream.py logic) with TinyStories train documents taken from the
beginning, until the story tokens reach the biography tokens. Seed 0.

Outputs (git-ignored): data/cache/streams/<run>.bin and .json; results/task4b/streams/<run>/.
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
from data.build_stream import (STREAM_DIR, TOK_DIR, interleave, load_people,  # noqa: E402
                               render_bios, tokens_per_step)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ft-B1k")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--people_set", default="B")
    p.add_argument("--n_people", type=int, default=1000)
    p.add_argument("--exposures", type=int, default=300)
    args = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    tps = tokens_per_step()

    people = load_people(args.people_set, args.n_people)
    bio_docs = render_bios(people, args.exposures, rng, tok, template5_even=True)
    n_bio = int(sum(len(d) for d in bio_docs))

    train = np.memmap(os.path.join(TOK_DIR, "stories_train.bin"), dtype=np.uint16, mode="r")
    offs = np.load(os.path.join(TOK_DIR, "stories_train_offsets.npy"))
    ends = np.append(offs[1:], len(train))
    n_story = int(np.searchsorted(ends, n_bio) + 1)  # docs until story tokens >= bio tokens
    story_docs = [train[offs[i]:ends[i]] for i in range(n_story)]
    n_story_tok = int(ends[n_story - 1])

    docs = interleave(story_docs, bio_docs, rng)
    n_tok = int(sum(len(d) for d in docs))
    meta = {"run": args.run, "seed": args.seed, "people_set": args.people_set,
            "n_people": len(people), "exposures": args.exposures, "template5_even": True,
            "n_story_docs": n_story, "story_tokens": n_story_tok, "n_bio_docs": len(bio_docs),
            "bio_tokens": n_bio, "mean_bio_tokens": n_bio / len(bio_docs),
            "tokens_per_step": tps, "stream_tokens": n_tok, "steps": n_tok // tps,
            "unused_tail_tokens": n_tok - (n_tok // tps) * tps}
    path = os.path.join(STREAM_DIR, f"{args.run}.bin")
    np.concatenate(docs).astype(np.uint16).tofile(path)
    assert os.path.getsize(path) == 2 * n_tok
    meta["wall_seconds"] = time.time() - t0
    with open(os.path.join(STREAM_DIR, f"{args.run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    res = os.path.join(ROOT, "results", "task4b", "streams", args.run)
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
