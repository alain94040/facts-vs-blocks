"""Shared helpers for revision 2 (hop/PLAN.md section 7): plain TransformerLens model, residual
states after every layer, the country-rank rule of Task 1.

The plain model is loaded exactly as in Task 0 step 2 (fold_ln, center_writing_weights and
center_unembed all False, float32 on MPS), so its residual stream is the model's own.
"""
import common
import torch


def load_tl():
    from transformer_lens import HookedTransformer
    tl = HookedTransformer.from_pretrained(common.MODEL, fold_ln=False, center_writing_weights=False,
                                           center_unembed=False, device=common.DEVICE, dtype=common.DTYPE)
    tl.eval()
    return tl


def resid_names(tl):
    return [f"blocks.{l}.hook_resid_post" for l in range(tl.cfg.n_layers)]


def run_states(tl, ids_l, positions):
    """Runs one prompt (list of ids). Returns (next-token probabilities at the last position,
    float64 CPU [vocab]; {name: residual stream after every layer at that position, float64 CPU
    [layers, d_model]} for each named position)."""
    names = resid_names(tl)
    with torch.no_grad():
        logits, cache = tl.run_with_cache(torch.tensor([ids_l], device=common.DEVICE),
                                          names_filter=lambda n: n in names)
    p = torch.softmax(logits[0, -1].cpu().double(), -1)
    st = {k: torch.stack([cache[n][0, pos] for n in names]).cpu().double() for k, pos in positions.items()}
    del cache, logits
    return p, st


def country_rank(p, ans, ctoks):
    """Task 1 country-rank rule: 1 + number of country first tokens with a strictly higher
    probability than the answer's (countries sharing a first token share its rank)."""
    pc = p[ctoks]
    return int((pc > p[ans]).sum()) + 1
