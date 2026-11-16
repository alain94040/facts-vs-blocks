"""Train the byte-level BPE tokenizer and tokenize TinyStories.

Tokenizer: vocab 4096, trained on all TinyStories train stories plus a sample of biographies
(training templates only, people from sets A and B). Saved to data/tokenizer.json.

Token streams (git-ignored): data/cache/tokens/stories_{train,validation}.bin as uint16, each
document preceded by <|endoftext|> (id 0), plus stories_{split}_offsets.npy holding the start
offset of every document (so later tasks can shuffle biographies in at the document level).

Writes results/task1/prepare/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_HOME = os.path.join(ROOT, "data", "cache", "hf")
os.environ.setdefault("HF_HOME", HF_HOME)  # must be set before importing datasets

import numpy as np  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

sys.path.insert(0, ROOT)
from data.make_bios import render_bio  # noqa: E402

EOT = "<|endoftext|>"
TOK_DIR = os.path.join(ROOT, "data", "cache", "tokens")


def load_stories():
    return load_dataset("roneneldan/TinyStories")


def bio_sample(n, seed):
    rng = np.random.default_rng(seed)
    people = []
    for s in ["A", "B"]:
        with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{s}.json")) as f:
            people += json.load(f)
    idx = rng.integers(len(people), size=n)
    return [render_bio(people[i], rng) for i in idx]


def train_tokenizer(stories, bios, vocab_size):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)

    def it():
        for i in range(0, len(stories), 10000):
            yield stories[i:i + 10000]
        for i in range(0, len(bios), 10000):
            yield bios[i:i + 10000]

    tok.train_from_iterator(it(), trainer=trainer, length=len(stories) + len(bios))
    return tok


def tokenize_split(tok, texts, name):
    eot = tok.token_to_id(EOT)
    chunks, offsets, pos = [], [], 0
    for i in range(0, len(texts), 20000):
        for enc in tok.encode_batch(texts[i:i + 20000]):
            ids = [eot] + enc.ids
            offsets.append(pos)
            pos += len(ids)
            chunks.append(np.asarray(ids, dtype=np.uint16))
    arr = np.concatenate(chunks)
    arr.tofile(os.path.join(TOK_DIR, f"stories_{name}.bin"))
    np.save(os.path.join(TOK_DIR, f"stories_{name}_offsets.npy"), np.asarray(offsets, np.int64))
    return len(arr), len(offsets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--n_bios", type=int, default=100000)
    args = p.parse_args()

    t0 = time.time()
    ds = load_stories()
    train = [t for t in ds["train"]["text"] if t.strip()]
    val = [t for t in ds["validation"]["text"] if t.strip()]
    bios = bio_sample(args.n_bios, args.seed)

    tok = train_tokenizer(train, bios, args.vocab_size)
    tok.save(os.path.join(ROOT, "data", "tokenizer.json"))
    assert tok.get_vocab_size() == args.vocab_size, tok.get_vocab_size()

    os.makedirs(TOK_DIR, exist_ok=True)
    n_train_tok, n_train_docs = tokenize_split(tok, train, "train")
    n_val_tok, n_val_docs = tokenize_split(tok, val, "validation")

    res = os.path.join(ROOT, "results", "task1", "prepare")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    metrics = {
        "seed": args.seed, "vocab_size": tok.get_vocab_size(), "n_bios_for_tokenizer": len(bios),
        "train_docs": n_train_docs, "train_tokens": int(n_train_tok),
        "validation_docs": n_val_docs, "validation_tokens": int(n_val_tok),
        "dropped_empty_train": len(ds["train"]) - len(train),
        "dropped_empty_validation": len(ds["validation"]) - len(val),
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
