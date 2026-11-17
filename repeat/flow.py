"""Experiment C: a forward pass that walks the layers itself, so interventions can be applied.

A layer maps x to y in two parts: x1 = x + A(x), y = x1 + M(x1); L(x) = y, u = y - x. For a block
of layers [start, end), L is the whole block. Interventions (repeat/PLAN.md section 2):

  {"kind": "base"}
  {"kind": "repeat",  "start": i, "end": j}            L(y)
  {"kind": "gain",    "start": i, "end": j, "g": g}    x + g*u
  {"kind": "partial", "start": i, "end": j, "a": a}    y + a*(L(y) - y)
  {"kind": "attn2",   "start": i}                      y + A(y)      (single layer)
  {"kind": "mlp2",    "start": i}                      y + M(y)      (single layer)
  {"kind": "swap",    "start": i, "j": j}              layer j's weights run at position i

The pieces A and M are the library's own submodules (input_layernorm + self_attn, and
post_attention_layernorm + mlp), with the library's rotary embeddings and causal mask, so `base`
reproduces model(input_ids).logits. Inputs are right-padded: with causal attention, real tokens
never attend to padding, so no padding mask is needed.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.masking_utils import (create_causal_mask,  # noqa: E402
                                        create_sliding_window_causal_mask)

DEV_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONFIRM_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Repeat task 5b only


def load(name, dtype=torch.float16, device="mps"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    return model.to(device).eval(), tok


class Flow:
    def __init__(self, model):
        self.model = model
        self.inner = model.model
        self.layers = self.inner.layers
        self.n_layers = len(self.layers)
        self.cfg = model.config

    # ---- pieces
    def _ctx(self, x, position_ids):
        kw = {"config": self.cfg, "inputs_embeds": x, "attention_mask": None,
              "past_key_values": None, "position_ids": position_ids}
        masks = {"full_attention": create_causal_mask(**kw)}
        if getattr(self.inner, "has_sliding_layers", False):
            masks["sliding_attention"] = create_sliding_window_causal_mask(**kw)
        pe = self.inner.rotary_emb(x, position_ids)
        return masks, pe, position_ids

    def A(self, l, x, ctx, pos=None):
        """Attention update of layer l's weights, run at position `pos` (default l)."""
        masks, pe, pid = ctx
        layer = self.layers[l]
        mask = masks[self.cfg.layer_types[pos if pos is not None else l]]
        h, _ = layer.self_attn(hidden_states=layer.input_layernorm(x), attention_mask=mask,
                               position_ids=pid, position_embeddings=pe)
        return h

    def M(self, l, x):
        layer = self.layers[l]
        return layer.mlp(layer.post_attention_layernorm(x))

    def layer(self, l, x, ctx, pos=None):
        x1 = x + self.A(l, x, ctx, pos)
        return x1 + self.M(l, x1)

    def block(self, start, end, x, ctx):
        for l in range(start, end):
            x = self.layer(l, x, ctx)
        return x

    # ---- forward
    @torch.no_grad()
    def hidden(self, input_ids, iv=None, stop_at=None):
        """Final hidden state (after the final norm) under intervention iv. With stop_at=k, return
        the residual stream at the input of layer k instead (no intervention applied before)."""
        iv = iv or {"kind": "base"}
        kind = iv["kind"]
        x = self.inner.embed_tokens(input_ids)
        pid = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        ctx = self._ctx(x, pid)
        if kind == "base":
            s, e = self.n_layers, self.n_layers
        else:
            s = iv["start"]
            e = iv.get("end", s + 1)
            assert 0 <= s < e <= self.n_layers
        x = self.block(0, s, x, ctx) if stop_at is None else self.block(0, min(s, stop_at), x, ctx)
        if stop_at is not None:
            assert stop_at <= s or kind == "base"
            return x
        if kind != "base":
            if kind == "repeat":
                x = self.block(s, e, self.block(s, e, x, ctx), ctx)
            elif kind == "gain":  # combinations in float32: exact for g = 1 and a in {0, 1}
                y = self.block(s, e, x, ctx)
                x = (x.float() + iv["g"] * (y.float() - x.float())).to(x.dtype)
            elif kind == "partial":
                y = self.block(s, e, x, ctx)
                y2 = self.block(s, e, y, ctx)
                x = (y.float() + iv["a"] * (y2.float() - y.float())).to(y.dtype)
            elif kind == "attn2":
                assert e == s + 1
                y = self.layer(s, x, ctx)
                x = y + self.A(s, y, ctx)
            elif kind == "mlp2":
                assert e == s + 1
                y = self.layer(s, x, ctx)
                x = y + self.M(s, y)
            elif kind == "swap":
                assert e == s + 1
                x = self.layer(iv["j"], x, ctx, pos=s)
            else:
                raise ValueError(kind)
            x = self.block(e, self.n_layers, x, ctx)
        return self.inner.norm(x)

    def logits(self, input_ids, iv=None):
        return self.model.lm_head(self.hidden(input_ids, iv))
