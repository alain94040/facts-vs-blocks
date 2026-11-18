"""Experiment E Part 2 shared code: two-hop items and batched forward passes.

Prompt format as in Experiment C (repeat/data.py): three worked example lines in front, then the
prompt; the answer starts with a space and its token ids come from repeat.data.tokenize_qa. Only
the answer's first token is scored (greedy: it is the arg-max at the last prompt token).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from middle.countries import EXAMPLES, ROWS  # noqa: E402
from repeat import data as D  # noqa: E402

TEMPLATES = {
    "HOP1": (lambda c, cap, cont, lang: f"The country whose capital is {cap} is", lambda c, cap, cont, lang: c),
    "HOP2-language": (lambda c, cap, cont, lang: f"The official language of {c} is",
                      lambda c, cap, cont, lang: lang),
    "HOP2-continent": (lambda c, cap, cont, lang: f"{c} is in the continent of",
                       lambda c, cap, cont, lang: cont),
    "TWO-language": (lambda c, cap, cont, lang:
                     f"The official language of the country whose capital is {cap} is",
                     lambda c, cap, cont, lang: lang),
    "TWO-continent": (lambda c, cap, cont, lang:
                      f"The country whose capital is {cap} is in the continent of",
                      lambda c, cap, cont, lang: cont),
}
TWO_KINDS = {"language": ("TWO-language", "HOP2-language"),
             "continent": ("TWO-continent", "HOP2-continent")}


def head(kind):
    p, a = TEMPLATES[kind]
    rows = [r for r in ROWS if r[0] in EXAMPLES]
    return "\n".join(f"{p(*r)} {a(*r)}." for r in rows) + "\n"


def items(tok):
    """{kind: [item]} for every non-example row; item = dict(country, prompt ids, answer first
    token, bridge first token, answer text)."""
    rows = [r for r in ROWS if r[0] not in EXAMPLES]
    bridge = {r[0]: D.tokenize_qa(tok, [("The country is", f" {r[0]}")])[0][1][0] for r in rows}
    out = {}
    for kind, (p, a) in TEMPLATES.items():
        h = head(kind)
        qa = D.tokenize_qa(tok, [(h + p(*r), f" {a(*r)}") for r in rows])
        out[kind] = [{"country": r[0], "capital": r[1], "continent": r[2], "language": r[3],
                      "answer": a(*r), "ids": pi, "ans": ai[0], "bridge": bridge[r[0]]}
                     for r, (pi, ai) in zip(rows, qa)]
    return out


def pad(seqs, pad_id=0):
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to("mps"), torch.tensor([len(s) - 1 for s in seqs], device="mps")


@torch.no_grad()
def states(flow, seqs, batch=32):
    """Residual stream at the last prompt token: (n, n_layers + 1, d) float32 on CPU; entry 0 is
    the embedding, entry l + 1 the stream after layer l (inputs right-padded; causal attention)."""
    out = []
    for b0 in range(0, len(seqs), batch):
        x, last = pad(seqs[b0:b0 + batch])
        h = flow.inner.embed_tokens(x)
        ctx = flow._ctx(h, torch.arange(h.shape[1], device=h.device)[None])
        r = torch.arange(len(last), device=h.device)
        st = [h[r, last].float().cpu()]
        for l in range(flow.n_layers):
            h = flow.layer(l, h, ctx)
            st.append(h[r, last].float().cpu())
        out.append(torch.stack(st, 1))
    return torch.cat(out)


@torch.no_grad()
def lens_logits(flow, s):
    """Logit lens: final norm and unembedding applied to residual states s (..., d), float32."""
    w = flow.inner.norm.weight
    return flow.model.lm_head(flow.inner.norm(s.to("mps").to(w.dtype))).float()


def greedy_ok(flow, seqs, answers, batch=32):
    st = states(flow, seqs, batch)
    top = lens_logits(flow, st[:, -1]).argmax(-1).cpu().numpy()
    return top == np.array(answers), st
