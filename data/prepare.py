"""Train the byte-level BPE tokenizer and tokenize TinyStories.

Tokenizer: vocab 4096, trained on all TinyStories train stories plus a sample of biographies
(training templates only, people from sets A and B). Saved to data/tokenizer.json.

Token streams (git-ignored): data/cache/tokens/stories_{train,validation}.bin as uint16, each
document preceded by <|endoftext|> (id 0), plus stories_{split}_offsets.npy holding the start
offset of every document (so later tasks can shuffle biographies in at the document level).

Writes results/task1/prepare/{metrics.json, cmd.txt}.

Task 2a (--entity_tokens): load the Task 1 tokenizer (data/tokenizer_base.json, ids 0..4095
unchanged), add each invented pool root word (data/make_bios.py:entity_words) as one dedicated
token with ids from 4096 up, assert that every pool word in every template encodes to exactly that
id (optionally preceded by a bare-space token) and round-trips, and save data/tokenizer.json.
Story streams are NOT changed (decided at Task 2a review): stories are re-tokenized with the base
tokenizer and asserted bit-identical to the Task 1 streams. Four pool words also occur as real
words in stories; the number of stories the new tokenizer would tokenize differently is reported.
Writes results/task2a/prepare/{metrics.json, cmd.txt}.
"""
import re
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
from data.make_bios import (ATTRIBUTES, TEMPLATES, entity_words,  # noqa: E402
                            render_bio)
from tokenizers import AddedToken  # noqa: E402

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


def tokenize_split(tok, texts, name, tag=""):
    eot = tok.token_to_id(EOT)
    chunks, offsets, pos = [], [], 0
    for i in range(0, len(texts), 20000):
        for enc in tok.encode_batch(texts[i:i + 20000]):
            ids = [eot] + enc.ids
            offsets.append(pos)
            pos += len(ids)
            chunks.append(np.asarray(ids, dtype=np.uint16))
    arr = np.concatenate(chunks)
    if tag == "check":  # compare with the existing stream, write nothing
        old = np.fromfile(os.path.join(TOK_DIR, f"stories_{name}.bin"), dtype=np.uint16)
        old_off = np.load(os.path.join(TOK_DIR, f"stories_{name}_offsets.npy"))
        assert np.array_equal(arr, old), f"{name}: story tokens differ from Task 1"
        assert np.array_equal(np.asarray(offsets), old_off), f"{name}: offsets differ"
        return len(arr), len(offsets)
    arr.tofile(os.path.join(TOK_DIR, f"stories_{name}.bin"))
    np.save(os.path.join(TOK_DIR, f"stories_{name}_offsets.npy"), np.asarray(offsets, np.int64))
    return len(arr), len(offsets)


ENTITY_SLOTS = {"cities": ["birth_city", "employer_city"], "universities": ["university"],
                "employers": ["employer"]}


def check_entity_tokens(tok, pools, words):
    """Every pool word, in every template it can appear in, encodes to its one dedicated id
    (a bare-space token in front is allowed) and the sentence round-trips."""
    vocab = tok.get_vocab()
    space = tok.encode(" ").ids
    assert len(space) == 1
    n_checked = 0

    def check(sentence, word):
        nonlocal n_checked
        enc = tok.encode(sentence)
        assert tok.decode(enc.ids) == sentence, sentence
        wid = vocab[word]
        assert enc.ids.count(wid) == 1, (sentence, word, enc.tokens)
        k = enc.ids.index(wid)
        lo, hi = enc.offsets[k]
        assert sentence[lo:hi] == word, (sentence, word)
        if lo > 0:  # preceded by a space, which must be its own bare token
            assert sentence[lo - 1] == " " and enc.ids[k - 1] == space[0], (sentence, enc.tokens)
        n_checked += 1

    other = {"name": f"{words['first_names'][0]} {words['last_names'][0]}"}
    filler = {"birth_date": "March 14, 1952", "birth_city": pools["cities"][0],
              "university": pools["universities"][0], "major": "geology",
              "employer": pools["employers"][0], "employer_city": pools["cities"][0]}
    for a in ATTRIBUTES:
        for tmpl in TEMPLATES[a]:
            for i, fn in enumerate(words["first_names"]):
                ln = words["last_names"][(i + 1) % len(words["last_names"])]
                s = tmpl.format(name=f"{fn} {ln}", v=filler[a])
                check(s, fn)
                check(s, ln)
    for pool, slots in ENTITY_SLOTS.items():
        for full, root in zip(pools[pool], words[pool]):
            for a in slots:
                for tmpl in TEMPLATES[a]:
                    check(tmpl.format(name=other["name"], v=full), root)
    return n_checked


def add_entity_tokens(args):
    t0 = time.time()
    base = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer_base.json"))
    assert base.get_vocab_size() == 4096
    with open(os.path.join(ROOT, "data", "cache", "bios", "pools.json")) as f:
        pools = json.load(f)
    words = entity_words(pools)
    flat = [w for ws in words.values() for w in ws]
    assert len(flat) == len(set(flat)) == 1550

    tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer_base.json"))
    n_added = tok.add_tokens([AddedToken(w, single_word=True) for w in flat])
    assert n_added == len(flat)
    vocab = tok.get_vocab()
    assert sorted(vocab[w] for w in flat) == list(range(4096, 4096 + len(flat)))
    base_vocab = base.get_vocab()
    assert all(vocab[k] == v for k, v in base_vocab.items())  # old ids unchanged
    n_checked = check_entity_tokens(tok, pools, words)
    tok.save(os.path.join(ROOT, "data", "tokenizer.json"))

    ds = load_stories()
    train = [t for t in ds["train"]["text"] if t.strip()]
    val = [t for t in ds["validation"]["text"] if t.strip()]
    n_train_tok, n_train_docs = tokenize_split(base, train, "train", tag="check")
    n_val_tok, n_val_docs = tokenize_split(base, val, "validation", tag="check")
    with open(os.path.join(ROOT, "results", "task1", "prepare", "metrics.json")) as f:
        t1 = json.load(f)
    assert (n_train_tok, n_val_tok) == (t1["train_tokens"], t1["validation_tokens"])

    # Stories the new tokenizer would encode differently (kept with base tokens, see docstring).
    word_re, flat_set = re.compile(r"[A-Za-z]+"), set(flat)
    differ = {}
    for split, texts in [("train", train), ("validation", val)]:
        cand = [(t, flat_set.intersection(word_re.findall(t))) for t in texts]
        cand = [(t, w) for t, w in cand if w]
        differ[split] = sum(tok.encode(t).ids != base.encode(t).ids for t, _ in cand)
        differ[split + "_words"] = sorted(set().union(*[w for _, w in cand]))

    res = os.path.join(ROOT, "results", "task2a", "prepare")
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "cmd.txt"), "w") as f:
        f.write(shlex.join([sys.executable] + sys.argv) + "\n")
    metrics = {
        "seed": args.seed, "base_vocab_size": base.get_vocab_size(),
        "vocab_size": tok.get_vocab_size(), "n_entity_tokens": n_added,
        "entity_tokens_by_pool": {k: len(v) for k, v in words.items()},
        "entity_id_range": [4096, 4096 + n_added - 1],
        "template_word_checks_passed": n_checked,
        "train_docs": n_train_docs, "train_tokens": int(n_train_tok),
        "validation_docs": n_val_docs, "validation_tokens": int(n_val_tok),
        "story_tokens_identical_to_task1": True,
        "stories_new_tokenizer_would_change": differ,
        "wall_seconds": time.time() - t0,
    }
    with open(os.path.join(res, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--n_bios", type=int, default=100000)
    p.add_argument("--entity_tokens", action="store_true", help="Task 2a mode, see docstring")
    args = p.parse_args()
    if args.entity_tokens:
        return add_entity_tokens(args)

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
