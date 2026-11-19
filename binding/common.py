"""Experiment F shared code (binding/PLAN.md): runs, people, contexts, measurement points,
quadruples, held-out data and the top-k transcoder.

Models in float32 on MPS; statistics on CPU in float64 where they are accumulated.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import common as ac  # noqa: E402
from data.make_bios import N_TRAIN_TEMPLATES, n_templates_for, render_bio  # noqa: E402
from train import evaluate as ev  # noqa: E402

DEVICE = ac.DEVICE
PILOT = "P2-main-2k-s0"
REAL = ["main-2k-s0", "main-2k-s1", "main-8k-s0", "main-8k-s1", "main-32k-s0", "main-32k-s1",
        "ctrl-8k-x4-s0", "fork-A-s0"]
CONTROL = "main-0-s0"
UNTRAINED = "untrained-s0"
N_LAYER, D_MODEL, D_MLP = ac.N_LAYER, 256, ac.D_MLP
POINTS = ["e"] + [f"{k}{l}" for l in range(N_LAYER) for k in ("a", "m")]
CONTEXTS = ["START", "MID"]
UNSEEN = (16000, 17000)          # set B people, the FORMAT people of Phase 2
RES = os.path.join(ROOT, "results", "binding")
CKPT = os.path.join(ROOT, "checkpoints", "binding")


def out_dir(task, run=None):
    d = os.path.join(RES, task) if run is None else os.path.join(RES, task, run)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- models and people

def load(run):
    if run == UNTRAINED:
        return ac.untrained_model(DEVICE, seed=0)
    return ac.load_model(run)


def meta(run):
    if run == UNTRAINED:
        return ac.run_meta("main-0-s0")
    return ac.run_meta(run)


def people(set_name):
    with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{set_name}.json")) as f:
        return json.load(f)


def trained_people(run):
    """The run's own people, in set order (empty for main-0 and the untrained model)."""
    m = meta(run)
    if not m.get("n_people"):
        return []
    return people(m["people_set"])[:m["n_people"]]


def unseen_people():
    return people("B")[UNSEEN[0]:UNSEEN[1]]


def pools():
    with open(os.path.join(ROOT, "data", "cache", "bios", "pools.json")) as f:
        return json.load(f)


_tok = {}


def tok_ids():
    if not _tok:
        t = ev.tokenizer()
        v = t.get_vocab()
        _tok.update({"tok": t, "vocab": v, "eot": v["<|endoftext|>"],
                     "space": t.encode(" ").ids[0],
                     "mid": t.encode("The employer of").ids})
    return _tok


def name_ids(first, last):
    t = tok_ids()
    return t["vocab"][first], t["vocab"][last]


def context_ids(ctx, first, last):
    """Prompt ending at the last-name token. START: <eot> F _ L; MID: <eot>The employer of F L."""
    t = tok_ids()
    f, l = name_ids(first, last)
    if ctx == "START":
        return [t["eot"], f, t["space"], l]
    return [t["eot"]] + t["mid"] + [t["space"], f, t["space"], l]


# ---------------------------------------------------------------- forward with measurement points

@torch.no_grad()
def points_at_last(model, ids, hooks=None, mlp_acts=False):
    """ids: (B, T) all of equal length (both contexts have fixed length). Returns the residual
    stream at the last position for every point, (B, 13, d), and optionally the post-GELU
    activations there, (B, 6, 1024). `hooks` is an analysis.common.Hooks instance (already
    entered) or None; capture is read from it when mlp_acts is set."""
    x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1], device=ids.device))
    out = [x[:, -1]]
    acts = []
    for blk in model.blocks:
        x = x + blk.attn(blk.ln1(x))
        out.append(x[:, -1])
        if mlp_acts:
            h = blk.ln2(x)
            pre = blk.mlp.up(h)
            g = torch.nn.functional.gelu(pre)
            acts.append(g[:, -1])
            x = x + blk.mlp.down(g)
        else:
            x = x + blk.mlp(blk.ln2(x))
        out.append(x[:, -1])
    r = torch.stack(out, 1)
    return (r, torch.stack(acts, 1)) if mlp_acts else r


# ---------------------------------------------------------------- quadruples

def all_person_names():
    return {p["name"] for p in people("A")} | {p["name"] for p in people("B")}


def quadruples(run, n=500, seed=0):
    """Real: F1 L1 = one of the run's people 0..999 (drawn without replacement), F2, L2 uniform
    from the pools until F1 L2, F2 L1, F2 L2 are no person of set A or B. Control: F1, L1 also
    uniform, with F1 L1 no person either. Returns {'real': [...], 'control': [...]} of
    (F1, L1, F2, L2)."""
    P = pools()
    fn, ln = P["first_names"], P["last_names"]
    names = all_person_names()
    tp = trained_people(run)[:1000]
    rng = np.random.default_rng(seed)
    out = {"real": [], "control": []}
    pick = rng.choice(len(tp), n, replace=False)
    for i in pick:
        f1, l1 = tp[i]["name"].split(" ")
        while True:
            f2, l2 = fn[rng.integers(len(fn))], ln[rng.integers(len(ln))]
            if f2 != f1 and l2 != l1 and not ({f"{f1} {l2}", f"{f2} {l1}", f"{f2} {l2}"} & names):
                out["real"].append((f1, l1, f2, l2))
                break
    while len(out["control"]) < n:
        f1, l1, f2, l2 = (fn[rng.integers(len(fn))], ln[rng.integers(len(ln))],
                          fn[rng.integers(len(fn))], ln[rng.integers(len(ln))])
        if f1 == f2 or l1 == l2:
            continue
        if {f"{f1} {l1}", f"{f1} {l2}", f"{f2} {l1}", f"{f2} {l2}"} & names:
            continue
        out["control"].append((f1, l1, f2, l2))
    return out


# ---------------------------------------------------------------- held-out data

N_HELD_ROWS = 3907          # 3,907 x 256 = 1,000,192 STORY positions (validation split)
N_REC_POS = 200_000         # recurrence positions: the first 200,000 of those


def heldout_story_rows():
    """Held-out STORY rows of 256 tokens: the first 3,907 rows of the validation split (never in
    any training stream; the transcoder stream uses the train split)."""
    return ev.val_rows(255)[:N_HELD_ROWS]


def heldout_bio_rows(run, n_rows, seed=12345):
    """Held-out biography rows of 256 tokens for the run's people 0..999: biographies rendered
    with training templates from a generator independent of the transcoder stream, packed."""
    tp = trained_people(run)[:1000]
    if not tp:
        return np.zeros((0, 256), dtype=np.int64)
    return pack_bios(tp, n_rows, np.random.default_rng(seed), meta(run).get("template5_even"),
                     list(range(len(tp))))


def pack_bios(tp, n_rows, rng, template5_even, index):
    t = tok_ids()
    rows, cur = [], []
    while len(rows) < n_rows:
        i = int(rng.integers(len(tp)))
        nt = n_templates_for(index[i]) if template5_even else N_TRAIN_TEMPLATES
        cur += [t["eot"]] + t["tok"].encode(render_bio(tp[i], rng, n_templates=nt)).ids
        while len(cur) >= 256 and len(rows) < n_rows:
            rows.append(cur[:256])
            cur = cur[256:]
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------- transcoder

class Transcoder(nn.Module):
    """Top-k transcoder (Gao et al. 2024): a = ReLU(topk(W_enc h + b_enc)), y = W_dec a + b_dec.
    W_dec is stored as (n_units, d) rows, i.e. W_dec[:, u] of the plan is dec[u]."""

    def __init__(self, n_units, k, d=D_MODEL):
        super().__init__()
        self.k = k
        self.dec = nn.Parameter(torch.randn(n_units, d) / d ** 0.5)
        self.renorm()
        self.enc = nn.Parameter(self.dec.detach().clone())       # W_enc = W_dec^T
        self.b_enc = nn.Parameter(torch.zeros(n_units))
        self.b_dec = nn.Parameter(torch.zeros(d))

    @torch.no_grad()
    def renorm(self):
        self.dec.div_(self.dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    def pre(self, h):
        return h @ self.enc.T + self.b_enc

    def encode(self, h):
        """Returns (values, indices) of the k kept units; values after ReLU."""
        v, i = self.pre(h).topk(self.k, dim=-1)
        return torch.relu(v), i

    def decode(self, v, i):
        return (v.unsqueeze(-1) * self.dec[i]).sum(-2) + self.b_dec

    def forward(self, h):
        v, i = self.encode(h)
        return self.decode(v, i)

    def dense(self, h):
        """Dense activations (..., n_units)."""
        v, i = self.encode(h)
        a = torch.zeros(*h.shape[:-1], self.dec.shape[0], device=h.device)
        return a.scatter(-1, i, v)


RECIPES = {"T1": {"n_units": 16384, "k": 32, "positions": 20_000_000},
           "T2": {"n_units": 32768, "k": 64, "positions": 40_000_000},
           # T3 (Revision 1): T2 + the auxiliary loss of Gao et al. 2024, A.2
           "T3": {"n_units": 32768, "k": 64, "positions": 40_000_000,
                  "aux_k": 512, "aux_coef": 1 / 32, "dead_positions": 10_000_000}}


def load_transcoders(run, tc=0, device=DEVICE):
    d = os.path.join(CKPT, run, f"tc{tc}")
    with open(os.path.join(d, "recipe.json")) as f:
        rec = json.load(f)
    out = []
    for l in range(N_LAYER):
        t = Transcoder(rec["n_units"], rec["k"])
        t.load_state_dict(torch.load(os.path.join(d, f"L{l}.pt"), map_location="cpu"))
        out.append(t.to(device).eval())
    return out, rec


class Replace:
    """Context manager: replace the MLP of the given layers by their transcoders."""

    def __init__(self, model, tcs, layers):
        self.model, self.tcs, self.layers, self.h = model, tcs, layers, []

    def __enter__(self):
        for l in self.layers:
            tc = self.tcs[l]
            self.h.append(self.model.blocks[l].mlp.register_forward_hook(
                lambda mod, args, out, tc=tc: tc(args[0])))
        return self

    def __exit__(self, *e):
        for h in self.h:
            h.remove()
