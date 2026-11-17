"""Experiment C scoring and statistics (repeat/PLAN.md section 4), and the Repeat task 0 driver.

score(flow, sets, split, iv) returns per-item arrays so that any two configurations can be
compared with a paired bootstrap (bootstrap_diff: 1,000 resamples over items, seed 0, 95%).

Repeat task 0:
  python repeat/evaluate.py --task0 --model dev  --seed 0     (development model; not results)
  python repeat/evaluate.py --task0 --model main --seed 0     (Qwen2.5-7B-Instruct)
"""
import argparse
import json
import os
import shlex
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from repeat import data as D  # noqa: E402
from repeat import flow as FL  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEVICE = "mps"
K_ORDER = {"start": 3, "up": [4, 5, 6], "down": [2]}
ACC_RANGE = (0.30, 0.70)
MODELS = {"dev": FL.DEV_MODEL, "main": FL.MAIN_MODEL, "confirm": FL.CONFIRM_MODEL}


def build_sets(tok, k):
    text, tinfo = D.text_chunks(tok)
    reason = D.tokenize_qa(tok, D.reason_items(k))
    rec_items = D.recall_items()
    recall = D.tokenize_qa(tok, rec_items)
    return {"TEXT": text, "REASON": reason, "RECALL": recall,
            "RECALL_kind": [x[2] for x in rec_items], "text_info": tinfo, "k": k,
            "splits": {s: D.split(len(v)) for s, v in [("TEXT", text), ("REASON", reason),
                                                        ("RECALL", recall)]}}


@torch.no_grad()
def _qa(flow, items, iv, batch):
    lp = np.zeros(len(items))
    ok = np.zeros(len(items), dtype=bool)
    ntok = 0
    order = np.argsort([len(p) + len(a) for p, a in items])
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        L = max(len(items[i][0]) + len(items[i][1]) for i in idx) - 1
        x = torch.zeros((len(idx), L), dtype=torch.long)
        rows, cols, tgt, owner = [], [], [], []
        for r, i in enumerate(idx):
            p, a = items[i]
            seq = p + a
            x[r, :len(seq) - 1] = torch.tensor(seq[:-1])
            for t, tok_id in enumerate(a):  # position len(p)-1+t predicts answer token t
                rows.append(r)
                cols.append(len(p) - 1 + t)
                tgt.append(tok_id)
                owner.append(i)
            ntok += len(seq) - 1
        h = flow.hidden(x.to(DEVICE), iv)
        logits = flow.model.lm_head(h[torch.tensor(rows), torch.tensor(cols)]).float()
        tg = torch.tensor(tgt, device=logits.device)
        l = torch.log_softmax(logits, -1).gather(1, tg[:, None])[:, 0].cpu().numpy()
        hit = (logits.argmax(-1) == tg).cpu().numpy()
        owner = np.array(owner)
        np.add.at(lp, owner, l)
        miss = np.zeros(len(items), dtype=bool)
        np.logical_or.at(miss, owner, ~hit)
        ok[idx] = ~miss[idx]
    return {"logprob": lp, "correct": ok}, ntok


@torch.no_grad()
def _text(flow, chunks, iv, batch):
    loss = np.zeros(len(chunks))
    for b0 in range(0, len(chunks), batch):
        x = torch.tensor(chunks[b0:b0 + batch]).to(DEVICE)
        h = flow.hidden(x[:, :-1], iv)
        for r in range(len(x)):
            lg = flow.model.lm_head(h[r]).float()
            loss[b0 + r] = F.cross_entropy(lg, x[r, 1:]).item()
    return {"loss": loss}, len(chunks) * (len(chunks[0]) - 1)


def score(flow, sets, split, iv=None, which=("TEXT", "REASON", "RECALL"), n_text=None,
          batch_qa=16, batch_text=4):
    """Per-item scores on the given split. n_text: use only the first n_text TEXT items of the
    split (Repeat task 4)."""
    out, ntok = {}, 0
    for s in which:
        idx = sets["splits"][s][split]
        if s == "TEXT" and n_text:
            idx = idx[:n_text]
        items = [sets[s][i] for i in idx]
        if s == "TEXT":
            r, n = _text(flow, items, iv, batch_text)
        else:
            r, n = _qa(flow, items, iv, batch_qa)
        r["index"] = idx
        out[s] = r
        ntok += n
    out["_tokens"] = ntok
    return out


def summarise(sc):
    d = {}
    if "TEXT" in sc:
        d["TEXT_loss"] = float(sc["TEXT"]["loss"].mean())
    for s in ("REASON", "RECALL"):
        if s in sc:
            d[f"{s}_logprob"] = float(sc[s]["logprob"].mean())
            d[f"{s}_acc"] = float(sc[s]["correct"].mean())
    return d


def bootstrap_diff(a, b, n=1000, seed=0):
    """Paired difference mean(a - b) with a 95% bootstrap interval over items."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def save_scores(path, sc):
    np.savez_compressed(path, **{f"{s}__{k}": v for s, r in sc.items() if isinstance(r, dict)
                                 for k, v in r.items()})


# ---------------------------------------------------------------- task 0

def exactness(flow, tok, dtype):
    text = ("The capital of France is Paris. Start with 7. Add 5. Multiply by 3. Subtract 4. "
            "The result is 32. The quick brown fox jumps over the lazy dog.")
    ids = torch.tensor([tok(text).input_ids] * 2).to(DEVICE)
    with torch.no_grad():
        ref = flow.model(ids).logits.float()
    d = lambda a, b: float((a.float() - b).abs().max().item())  # noqa: E731
    L = flow.n_layers // 2
    res = {"dtype": str(dtype), "base_vs_library": d(flow.logits(ids), ref),
           "gain1_vs_base": d(flow.logits(ids, {"kind": "gain", "start": L, "g": 1.0}), ref),
           "partial0_vs_base": d(flow.logits(ids, {"kind": "partial", "start": L, "a": 0.0}), ref),
           "partial1_vs_repeat": d(flow.logits(ids, {"kind": "partial", "start": L, "a": 1.0}),
                                   flow.logits(ids, {"kind": "repeat", "start": L}).float()),
           "swap_ii_vs_base": d(flow.logits(ids, {"kind": "swap", "start": L, "j": L}), ref),
           "block_gain1_vs_base": d(flow.logits(ids, {"kind": "gain", "start": L - 2,
                                                       "end": L + 2, "g": 1.0}), ref)}
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    res["tolerance"] = tol
    res["pass"] = all(v < tol for k, v in res.items() if k.endswith(("_library", "_base",
                                                                       "_repeat")))
    return res


def task0(args):
    out = os.path.join(ROOT, "results", "repeat", "task0", args.model)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    torch.manual_seed(args.seed)
    name = MODELS[args.model]
    res = {"model": name, "seed": args.seed}
    t0 = time.time()
    dtype = torch.float16
    if args.model == "dev":  # exactness in float32 on the development model, then float16
        m, tok = FL.load(name, dtype=torch.float32)
        res["exactness_float32"] = exactness(FL.Flow(m), tok, torch.float32)
        print("exactness float32", res["exactness_float32"], flush=True)
        del m
    m, tok = FL.load(name, dtype=dtype)
    flow = FL.Flow(m)
    res["load_seconds"] = time.time() - t0
    res["n_layers"] = flow.n_layers
    res["exactness_float16"] = exactness(flow, tok, dtype)
    print("exactness float16", res["exactness_float16"], flush=True)

    # difficulty: choose k on REASON SEARCH accuracy
    tried, k, chosen = [], K_ORDER["start"], None
    sets = build_sets(tok, k)
    res["recall_counts"] = {"capital": sets["RECALL_kind"].count("capital"),
                            "element": sets["RECALL_kind"].count("element")}
    res["text_info"] = sets["text_info"]
    t1 = time.time()
    sc = score(flow, sets, "SEARCH")
    dt = time.time() - t1
    res["speed"] = {"tokens": sc["_tokens"], "seconds": dt, "tokens_per_second": sc["_tokens"] / dt,
                    "seconds_per_full_configuration_search": dt}
    print("speed", res["speed"], flush=True)
    acc = float(sc["REASON"]["correct"].mean())
    tried.append({"k": k, "reason_search_acc": acc})
    print("k", k, "acc", acc, flush=True)
    if ACC_RANGE[0] <= acc <= ACC_RANGE[1]:
        chosen = k
    else:
        for k2 in (K_ORDER["up"] if acc > ACC_RANGE[1] else K_ORDER["down"]):
            s2 = build_sets(tok, k2)
            a2 = float(score(flow, s2, "SEARCH", which=("REASON",))["REASON"]["correct"].mean())
            tried.append({"k": k2, "reason_search_acc": a2})
            print("k", k2, "acc", a2, flush=True)
            if ACC_RANGE[0] <= a2 <= ACC_RANGE[1]:
                chosen, sets = k2, s2
                break
    res["difficulty"] = {"tried": tried, "chosen_k": chosen, "range": ACC_RANGE}
    if chosen is not None:
        sc = score(flow, sets, "SEARCH")
        res["baseline_search"] = summarise(sc)
        save_scores(os.path.join(out, "baseline_search.npz"), sc)
        res["baseline_search_by_kind"] = {
            kind: float(np.mean([c for c, i in zip(sc["RECALL"]["correct"], sc["RECALL"]["index"])
                                 if sets["RECALL_kind"][i] == kind]))
            for kind in ("capital", "element")}
    res["wall_seconds"] = time.time() - t0
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task0", action="store_true")
    p.add_argument("--model", choices=list(MODELS), default="dev")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.task0:
        task0(args)


if __name__ == "__main__":
    main()
