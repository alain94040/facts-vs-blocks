"""Hop task 0, extra check (reported, not gated): TransformerLens on MPS warns that it "may produce
silently incorrect results" with PyTorch 2.15.0. Gate H0 step 2 compares two TransformerLens models
on MPS, which would not catch that. This compares final-position log-probs of plain transformers
(Gemma2ForCausalLM, float32, eager attention) on CPU and on MPS with the ReplacementModel on MPS, on the
same 6 TWO prompts as Task 0 (TARGET + 5 CAP, seed 0).
Usage: python hop/task0_hf.py --seed 0
Writes results/hop/task0/hf_check/{metrics.json, cmd.txt}.
"""
import argparse
import json
import os
import random

import common
import data
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task0", "hf_check")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    its = data.items(tok)
    chosen = [its[0]] + random.Random(args.seed).sample([it for it in its if it["set"] == "CAP"], 5)
    ids = [torch.tensor([common.tokenize(tok, it["TWO"])]) for it in chosen]
    lp = {}
    for dev in ("cpu", "mps"):
        hf = AutoModelForCausalLM.from_pretrained(common.MODEL, dtype=torch.float32,
                                                  attn_implementation="eager").to(dev).eval()
        with torch.no_grad():
            lp["hf_" + dev] = [torch.log_softmax(hf(x.to(dev)).logits[0, -1].cpu().double(), -1) for x in ids]
        del hf
        common.free()
    rm = common.load_replacement_model()
    with torch.no_grad():
        lp["rm_mps"] = [torch.log_softmax(rm(x.to(common.DEVICE))[0, -1].cpu().double(), -1) for x in ids]

    def cmp(a, b):
        return {"max_abs_diff": max(float((x - y).abs().max()) for x, y in zip(lp[a], lp[b])),
                "top1_agree": sum(int(x.argmax() == y.argmax()) for x, y in zip(lp[a], lp[b]))}
    m = {"seed": args.seed, "prompts": [it["id"] for it in chosen],
         "hf_cpu_vs_hf_mps": cmp("hf_cpu", "hf_mps"), "hf_cpu_vs_rm_mps": cmp("hf_cpu", "rm_mps"),
         "hf_mps_vs_rm_mps": cmp("hf_mps", "rm_mps"),
         "top1": {k: [tok.decode([int(x.argmax())]) for x in v] for k, v in lp.items()}}
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
