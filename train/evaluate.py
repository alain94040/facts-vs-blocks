"""Task 2 evaluations.

- Story validation loss (nats/token): validation stream cut into non-overlapping rows of
  seq_len+1 tokens; mean loss over every predicted token. `val_sample_loss` uses a fixed random
  2,000 rows (seed 0) for the in-training curve.
- Fact recall (exact match) and fact loss. Prompt = <|endoftext|> + sentence prefix with trailing
  space removed. Expected continuation = ids of the full sentence minus the prefix ids, i.e. the
  ids of " " + answer + ".". Greedy decoding reproduces the expected ids exactly iff, under teacher
  forcing on the expected ids, the argmax at every answer position equals the expected token; we
  compute it that way in one forward pass. Fact loss = mean teacher-forced loss per answer token.
  Reported per attribute and overall, for training templates (1-4) and held-out template 5,
  on set A and set B.
- Induction: 1,000 sequences of 50 random tokens (ids 1..vocab-1, i.e. excluding <|endoftext|>)
  repeated twice. Ratio = mean loss on tokens 2..50 of the second copy / mean loss on tokens 2..50
  of the first copy (the first token of each copy is unpredictable from context and excluded).

All evaluation sets are fixed (seed 0) and independent of the run's training seed.

Standalone: python train/evaluate.py --ckpt <path> --run <stream name> --task task2 --run_name X --seed 0
"""
import argparse
import json
import os
import shlex
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.make_bios import (ATTRIBUTES, HELDOUT_TEMPLATE, N_TRAIN_TEMPLATES,  # noqa: E402
                            TEMPLATES)
from model.gpt import GPT, GPTConfig  # noqa: E402

EVAL_SEED = 0
N_RECALL_PEOPLE = 8000
N_VAL_SAMPLE = 2000

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = Tokenizer.from_file(os.path.join(ROOT, "data", "tokenizer.json"))
    return _tok


def val_rows(seq_len):
    data = np.fromfile(os.path.join(ROOT, "data", "cache", "tokens", "stories_validation.bin"),
                       dtype=np.uint16)
    n = len(data) // (seq_len + 1)
    return data[:n * (seq_len + 1)].reshape(n, seq_len + 1).astype(np.int64)


def val_sample_rows(seq_len):
    rows = val_rows(seq_len)
    idx = np.sort(np.random.default_rng(EVAL_SEED).choice(len(rows), N_VAL_SAMPLE, replace=False))
    return rows[idx]


@torch.no_grad()
def story_loss(model, rows, device, batch=128):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(rows[i:i + batch]).to(device)
        _, loss = model(x[:, :-1], x[:, 1:], reduction="sum")
        tot += loss.item()
        n += x[:, 1:].numel()
    model.train()
    return tot / n


def eval_people(people_set, run_meta):
    """People to test on set A or B, as in PLAN.md Task 2."""
    with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{people_set}.json")) as f:
        people = json.load(f)
    n = N_RECALL_PEOPLE
    if run_meta.get("people_set") == people_set and run_meta.get("n_people", 0) > 0:
        n = run_meta["n_people"]
    people = people[:n]
    if len(people) > N_RECALL_PEOPLE:
        idx = np.sort(np.random.default_rng(EVAL_SEED).choice(len(people), N_RECALL_PEOPLE,
                                                              replace=False))
        people = [people[i] for i in idx]
    return people


def build_prompts(people):
    """Return list of (attribute, template_idx, prompt_ids, answer_ids)."""
    tok = tokenizer()
    eot = tok.token_to_id("<|endoftext|>")
    out = []
    for p in people:
        for a in ATTRIBUTES:
            for t in range(len(TEMPLATES[a])):
                tmpl = TEMPLATES[a][t]
                full = tmpl.format(name=p["name"], v=p[a])
                prefix = tmpl.split("{v}")[0].format(name=p["name"]).rstrip(" ")
                full_ids = [eot] + tok.encode(full).ids
                pre_ids = [eot] + tok.encode(prefix).ids
                assert full_ids[:len(pre_ids)] == pre_ids, (full, prefix)
                out.append((a, t, pre_ids, full_ids[len(pre_ids):]))
    return out


@torch.no_grad()
def fact_eval(model, prompts, device, batch=512):
    model.eval()
    order = np.argsort([len(p) + len(a) for _, _, p, a in prompts])
    correct = np.zeros(len(prompts), dtype=bool)
    loss_sum = np.zeros(len(prompts))
    n_ans = np.array([len(a) for _, _, _, a in prompts])
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        L = max(len(prompts[j][2]) + len(prompts[j][3]) for j in idx)
        x = torch.zeros((len(idx), L), dtype=torch.long)
        y = torch.full((len(idx), L), -1, dtype=torch.long)
        for r, j in enumerate(idx):
            _, _, p, a = prompts[j]
            seq = p + a
            x[r, :len(seq)] = torch.tensor(seq)
            # position k predicts token k+1; answer tokens sit at len(p) .. len(seq)-1
            y[r, len(p) - 1:len(seq) - 1] = torch.tensor(a)
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        logits = logits.float()
        mask = y >= 0
        tl = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1).clamp(min=0),
                             reduction="none").view(y.shape) * mask
        hit = (logits.argmax(-1) == y) | ~mask
        correct[idx] = hit.all(dim=1).cpu().numpy()
        loss_sum[idx] = tl.sum(dim=1).cpu().numpy()
    model.train()

    attrs = np.array([a for a, _, _, _ in prompts])
    tmpl = np.array([t for _, t, _, _ in prompts])
    res = {}
    for split, sel in [("train_templates", tmpl < N_TRAIN_TEMPLATES),
                       ("heldout_template", tmpl == HELDOUT_TEMPLATE)]:
        d = {"recall": float(correct[sel].mean()),
             "fact_loss": float(loss_sum[sel].sum() / n_ans[sel].sum()),
             "per_attribute": {}}
        for a in ATTRIBUTES:
            s = sel & (attrs == a)
            d["per_attribute"][a] = {"recall": float(correct[s].mean()),
                                     "fact_loss": float(loss_sum[s].sum() / n_ans[s].sum())}
        res[split] = d
    return res


@torch.no_grad()
def induction_ratio(model, device, vocab_size, n=1000, length=50):
    model.eval()
    g = np.random.default_rng(EVAL_SEED)
    seq = g.integers(1, vocab_size, size=(n, length))
    x = torch.from_numpy(np.concatenate([seq, seq], axis=1)).to(device)
    tot = torch.zeros(2 * length - 1, device=device)
    for i in range(0, n, 250):
        xb = x[i:i + 250]
        logits, _ = model(xb[:, :-1])
        tl = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), xb[:, 1:].reshape(-1),
                             reduction="none").view(xb.shape[0], -1)
        tot += tl.sum(0)
    per_pos = (tot / n).cpu().numpy()  # per_pos[k] = loss predicting token k+1
    first = per_pos[0:length - 1].mean()            # tokens 2..50 of copy 1
    second = per_pos[length:2 * length - 1].mean()  # tokens 2..50 of copy 2
    model.train()
    return {"loss_first_copy": float(first), "loss_second_copy": float(second),
            "ratio": float(second / first)}


def full_eval(model, run_meta, device):
    seq_len = model.cfg.ctx_len
    out = {"story_val_loss": story_loss(model, val_rows(seq_len), device),
           "story_val_sample_loss": story_loss(model, val_sample_rows(seq_len), device)}
    for s in ["A", "B"]:
        people = eval_people(s, run_meta)
        r = fact_eval(model, build_prompts(people), device)
        r["n_people"] = len(people)
        out[f"facts_set_{s}"] = r
    out["induction"] = induction_ratio(model, device, model.cfg.vocab_size)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--run", required=True, help="stream name, e.g. main-2k")
    p.add_argument("--task", default="task2")
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu")
    model = GPT(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    with open(os.path.join(ROOT, "data", "cache", "streams", f"{args.run}.json")) as f:
        meta = json.load(f)
    out_dir = os.path.join(ROOT, "results", args.task, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    res = {"ckpt": args.ckpt, "step": ck["step"], **full_eval(model, meta, device)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
