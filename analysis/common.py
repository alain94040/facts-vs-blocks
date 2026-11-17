"""Shared Phase 2 machinery: models, the three datasets, components, ablation and importance.

Datasets (PLAN.md Phase 2, all fixed with seed 0, shared by every model):
  STORY  : the 2,000 fixed validation rows of Task 2 (evaluate.val_sample_rows), loss on all tokens.
  FACT   : recall prompts, templates 1-4, first 1,000 people of the run's own set (set A for runs
           without biographies; set B for fork-B). Loss on the answer tokens; recall = exact match.
  FORMAT : one biography each of set B people 16,000-16,999 (same rendering as the Task 2 name-copy
           test, seed 0), prefixed by <|endoftext|>. Loss on every token except attribute value
           tokens and the three tokens of the first mention of the name.

Components:
  neuron (l, i): input i of blocks[l].mlp.down, i.e. post-GELU activation (6 x 1,024).
  head (l, h): slice 32h:32h+32 of the input of blocks[l].attn.out (6 x 8).
Ablation replaces a component's activation at every position by its mean over the reference sample
(the first 1,000 STORY rows plus the FORMAT biographies, padding excluded).
Importance I_D(c) = | sum over positions of (mean_c - a_c) . dLbar/da_c |, with Lbar the mean loss
over D's scored tokens: a first-order estimate of the change in Lbar when c is mean-ablated.
"""
import copy
import glob
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from data.make_bios import ATTRIBUTES, N_TRAIN_TEMPLATES, TEMPLATES, render_bio  # noqa: E402
from model.gpt import GPT, GPTConfig  # noqa: E402
from train import evaluate as ev  # noqa: E402

SEED = 0
N_FACT_PEOPLE = 1000
N_REF_STORY = 1000
N_LAYER, N_HEAD, D_HEAD, D_MLP = 6, 8, 32, 1024
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PILOT = {"P2-main-2k-s0": ("task2a", "P2-main-2k"), "P1-main-2k-s0": ("task2a", "P1-main-2k")}


# ---------------------------------------------------------------- models

def stream_of(run):
    if run in PILOT:
        return PILOT[run][1]
    return run.rsplit("-s", 1)[0]


def ckpt_path(run):
    task = PILOT[run][0] if run in PILOT else "task2"
    files = glob.glob(os.path.join(ROOT, "checkpoints", task, run, "step_*.pt"))
    assert files, run
    return max(files, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))


def run_meta(run):
    with open(os.path.join(ROOT, "data", "cache", "streams", f"{stream_of(run)}.json")) as f:
        return json.load(f)


def load_model(run, device=DEVICE):
    ck = torch.load(ckpt_path(run), map_location="cpu")
    assert ck["step"] == ck["total_steps"], (run, ck["step"], ck["total_steps"])
    model = GPT(GPTConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def untrained_model(device=DEVICE, seed=0):
    """The initial weights: same config and init as training with --seed 0."""
    with open(os.path.join(ROOT, "configs", "model_small.json")) as f:
        cfg = json.load(f)
    torch.manual_seed(seed)
    return GPT(GPTConfig(**cfg)).to(device).eval()


# ---------------------------------------------------------------- datasets

_cache = {}


def story_rows():
    if "story" not in _cache:
        _cache["story"] = ev.val_sample_rows(256)
    return _cache["story"]


def fact_set(meta):
    return meta.get("people_set") or "A"


def fact_prompts(meta):
    """(attribute, template, prompt ids, answer ids, person index); templates 1-4."""
    s = fact_set(meta)
    key = ("fact", s)
    if key not in _cache:
        with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{s}.json")) as f:
            people = json.load(f)[:N_FACT_PEOPLE]
        pr = ev.build_prompts(list(enumerate(people)))
        _cache[key] = [p for p in pr if p[1] < N_TRAIN_TEMPLATES]
    return _cache[key]


def format_bios():
    """[(ids, loss mask over targets)]: mask[k] says whether predicting ids[k+1] is scored."""
    if "format" in _cache:
        return _cache["format"]
    tok = ev.tokenizer()
    eot = tok.token_to_id("<|endoftext|>")
    with open(os.path.join(ROOT, "data", "cache", "bios", "people_B.json")) as f:
        people = json.load(f)[ev.NAME_COPY_PEOPLE[0]:ev.NAME_COPY_PEOPLE[1]]
    rng = np.random.default_rng(ev.EVAL_SEED)  # identical to the Task 2 name-copy rendering
    out = []
    for p in people:
        twin = copy.deepcopy(rng)
        text = render_bio(p, rng)
        # replay render_bio's draws to get each sentence, so value spans are exact even when two
        # attributes share a value (e.g. birth city = employer city)
        sents = []
        for a in ATTRIBUTES:
            t = int(twin.integers(N_TRAIN_TEMPLATES))
            sents.append((a, TEMPLATES[a][t]))
        order = twin.permutation(len(sents))
        excl, pos = [], 0
        for k in order:
            a, tmpl = sents[k]
            sent = tmpl.format(name=p["name"], v=p[a])
            vs = pos + len(tmpl.split("{v}")[0].format(name=p["name"]))
            assert text[vs:vs + len(p[a])] == p[a]
            excl.append((vs, vs + len(p[a])))
            pos += len(sent) + 1
        assert pos - 1 == len(text)
        j = text.find(p["name"])  # first mention of the name
        excl.append((j, j + len(p["name"])))
        enc = tok.encode(text)
        keep = [not any(s < e2 and e > s2 for s2, e2 in excl) for (s, e) in enc.offsets]
        ids = [eot] + enc.ids
        out.append((ids, np.array(keep, dtype=bool)))  # target k+1 <-> enc token k
    _cache["format"] = out
    return out


def ref_batches(batch=128, story_only=False):
    """Reference sample for means: first 1,000 STORY rows, then the FORMAT biographies (skipped
    with story_only, used for main-0 in Task 5). Yields (x, valid mask over positions)."""
    rows = story_rows()[:N_REF_STORY, :-1]
    for i in range(0, len(rows), batch):
        x = torch.from_numpy(rows[i:i + batch])
        yield x, torch.ones_like(x, dtype=torch.bool)
    if story_only:
        return
    bios = format_bios()
    for i in range(0, len(bios), batch):
        chunk = bios[i:i + batch]
        L = max(len(ids) for ids, _ in chunk)
        x = torch.zeros((len(chunk), L), dtype=torch.long)
        m = torch.zeros((len(chunk), L), dtype=torch.bool)
        for r, (ids, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
            m[r, :len(ids)] = True
        yield x, m


def dataset_batches(name, meta, batch=None):
    """Yield (x, y) with y = -1 on unscored targets, for STORY, FACT or FORMAT."""
    if name == "STORY":
        rows = story_rows()
        b = batch or 64
        for i in range(0, len(rows), b):
            r = torch.from_numpy(rows[i:i + b])
            yield r[:, :-1], r[:, 1:]
    elif name == "FACT":
        pr = fact_prompts(meta)
        b = batch or 1024
        for i in range(0, len(pr), b):
            chunk = pr[i:i + b]
            L = max(len(p) + len(a) for _, _, p, a, _ in chunk) - 1
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            y = torch.full((len(chunk), L), -1, dtype=torch.long)
            for r, (_, _, p, a, _) in enumerate(chunk):
                seq = p + a
                x[r, :len(seq) - 1] = torch.tensor(seq[:-1])
                y[r, len(p) - 1:len(seq) - 1] = torch.tensor(a)
            yield x, y
    elif name == "FORMAT":
        bios = format_bios()
        b = batch or 250
        for i in range(0, len(bios), b):
            chunk = bios[i:i + b]
            L = max(len(ids) for ids, _ in chunk) - 1
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            y = torch.full((len(chunk), L), -1, dtype=torch.long)
            for r, (ids, keep) in enumerate(chunk):
                x[r, :len(ids) - 1] = torch.tensor(ids[:-1])
                t = torch.tensor(ids[1:])
                t[~torch.from_numpy(keep)] = -1
                y[r, :len(ids) - 1] = t
            yield x, y
    else:
        raise ValueError(name)


def n_scored(name, meta):
    if ("n", name, fact_set(meta)) not in _cache:
        _cache[("n", name, fact_set(meta))] = sum(int((y >= 0).sum())
                                                  for _, y in dataset_batches(name, meta))
    return _cache[("n", name, fact_set(meta))]


# ---------------------------------------------------------------- hooks

class Hooks:
    """Context manager installing pre-hooks on every mlp.down and attn.out.

    ablate_neurons: {layer: LongTensor of neuron indices}; ablate_heads: {layer: list of heads}.
    Ablated activations are replaced by `means` (from component_means).
    capture=True keeps each hooked input (retaining its grad) in self.acts[('mlp'|'attn', l)].
    """

    def __init__(self, model, means=None, ablate_neurons=None, ablate_heads=None, capture=False):
        self.model, self.means, self.capture = model, means, capture
        self.an = ablate_neurons or {}
        self.ah = ablate_heads or {}
        self.acts, self.handles = {}, []

    def _make(self, kind, l):
        def hook(mod, args):
            a = args[0]
            if kind == "mlp" and l in self.an and len(self.an[l]):
                idx = self.an[l]
                a = a.clone()
                a[..., idx.to(a.device)] = self.means["mlp"][l][idx].to(a.device)
            if kind == "attn" and l in self.ah and len(self.ah[l]):
                a = a.clone()
                for h in self.ah[l]:
                    a[..., h * D_HEAD:(h + 1) * D_HEAD] = \
                        self.means["attn"][l][h * D_HEAD:(h + 1) * D_HEAD].to(a.device)
            if self.capture:
                if a.requires_grad:
                    a.retain_grad()
                self.acts[(kind, l)] = a
            return (a,) if a is not args[0] else None
        return hook

    def __enter__(self):
        for l, blk in enumerate(self.model.blocks):
            self.handles.append(blk.mlp.down.register_forward_pre_hook(self._make("mlp", l)))
            self.handles.append(blk.attn.out.register_forward_pre_hook(self._make("attn", l)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def component_means(model, story_only=False):
    """Mean of every mlp.down input and attn.out input over the reference sample (float64)."""
    sums = {"mlp": [torch.zeros(D_MLP, dtype=torch.float64) for _ in range(N_LAYER)],
            "attn": [torch.zeros(N_HEAD * D_HEAD, dtype=torch.float64) for _ in range(N_LAYER)]}
    n = 0
    with Hooks(model, capture=True) as hk:
        for x, m in ref_batches(story_only=story_only):
            model(x.to(DEVICE))
            mm = m.to(DEVICE)
            for kind in ("mlp", "attn"):
                for l in range(N_LAYER):
                    a = hk.acts[(kind, l)][mm]
                    sums[kind][l] += a.double().sum(0).cpu() if DEVICE.type == "cpu" else \
                        a.float().sum(0).cpu().double()
            n += int(m.sum())
    return {k: [(s / n).float() for s in v] for k, v in sums.items()}, n


def importance(model, means, name, meta):
    """I_D for all neurons (6 x 1024) and heads (6 x 8), as numpy float64 arrays."""
    N = n_scored(name, meta)
    tot_n = torch.zeros(N_LAYER, D_MLP, dtype=torch.float64)
    tot_h = torch.zeros(N_LAYER, N_HEAD, dtype=torch.float64)
    mu = {k: [m.to(DEVICE) for m in v] for k, v in means.items()}
    for x, y in dataset_batches(name, meta):
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad(set_to_none=True)
        with Hooks(model, capture=True) as hk:
            _, loss = model(x, y, reduction="sum")
            (loss / N).backward()
        valid = (x != 0) | (y >= 0)  # padding: id 0 with no target; story rows never pad
        if name == "STORY":
            valid = torch.ones_like(valid)
        else:
            # positions up to the last scored target are real tokens; later ones are padding
            last = torch.where(y >= 0, torch.arange(y.shape[1], device=DEVICE), -1).max(1).values
            valid = torch.arange(y.shape[1], device=DEVICE)[None, :] <= last[:, None]
        vm = valid[..., None].float()
        for l in range(N_LAYER):
            a = hk.acts[("mlp", l)]
            c = ((mu["mlp"][l] - a.detach()) * a.grad * vm).sum((0, 1))
            tot_n[l] += c.cpu().double()
            a = hk.acts[("attn", l)]
            c = ((mu["attn"][l] - a.detach()) * a.grad * vm).sum((0, 1))
            tot_h[l] += c.view(N_HEAD, D_HEAD).sum(1).cpu().double()
    model.zero_grad(set_to_none=True)
    return np.abs(tot_n.numpy()), np.abs(tot_h.numpy())


# ---------------------------------------------------------------- evaluations

@torch.no_grad()
def mean_loss(model, name, meta):
    tot, n = 0.0, 0
    for x, y in dataset_batches(name, meta):
        _, loss = model(x.to(DEVICE), y.to(DEVICE), reduction="sum")
        tot += loss.item()
        n += int((y >= 0).sum())
    return tot / n


@torch.no_grad()
def fact_recall(model, meta):
    """Exact-match recall (teacher-forced argmax over all answer tokens) and fact loss on FACT."""
    hits, tot, n = 0, 0.0, 0
    for x, y in dataset_batches("FACT", meta):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x)
        mask = y >= 0
        ok = (logits.argmax(-1) == y) | ~mask
        hits += int(ok.all(1).sum())
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-1, reduction="sum").item()
        n += int(mask.sum())
    return hits / len(fact_prompts(meta)), tot / n


def evaluate_all(model, meta):
    """The four Task 3 measurements, for the model as currently hooked."""
    rec, fl = fact_recall(model, meta)
    nc = ev.name_copy(model, DEVICE, meta)
    model.eval()
    return {"recall": rec, "fact_loss": fl, "format_loss": mean_loss(model, "FORMAT", meta),
            "name_copy": nc["accuracy"], "story_loss": mean_loss(model, "STORY", meta)}


def neuron_sets(flat_idx):
    """Flat neuron indices (l * 1024 + i) -> {layer: LongTensor}."""
    flat_idx = np.asarray(flat_idx)
    return {l: torch.from_numpy(flat_idx[flat_idx // D_MLP == l] % D_MLP).long()
            for l in range(N_LAYER)}
