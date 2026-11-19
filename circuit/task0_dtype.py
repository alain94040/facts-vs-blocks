"""Circuit task 0, dtype check: final-position log-prob differences between plain transformers (HF)
and the TransformerLens model circuit-tracer wraps (TL, loaded as ReplacementModel loads it:
fold_ln=False, center_writing_weights=False, center_unembed=False), in float32 and bfloat16.

The ReplacementModel's forward pass runs the original MLPs (transcoders are only used for
attribution), so the TL logits are the circuit-tracer model's logits.

Usage: python circuit/task0_dtype.py --which dev --n 20 --seed 0
Writes results/circuit/task0/dtype_<which>/{metrics.json, cmd.txt}.
"""
import argparse
import gc
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import tasks  # noqa: E402
from task0 import MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[args.which][0]
    out = os.path.join(ROOT, "results", "circuit", "task0", f"dtype_{args.which}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    tok = AutoTokenizer.from_pretrained(name)
    seeds = list(range(9001, 9001 + args.n))
    ids = [tok(tasks.make("PROG", "P1", s)["prompt"], return_tensors="pt")["input_ids"] for s in seeds]
    lp = {}
    for dt in ("float32", "bfloat16"):
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dt)).to("mps").eval()
        with torch.no_grad():
            lp["hf", dt] = [torch.log_softmax(hf(x.to("mps")).logits[0, -1].float(), -1).cpu() for x in ids]
        del hf
        gc.collect()
        torch.mps.empty_cache()
        tl = HookedTransformer.from_pretrained(name, fold_ln=False, center_writing_weights=False,
                                               center_unembed=False, device="mps", dtype=getattr(torch, dt))
        with torch.no_grad():
            lp["tl", dt] = [torch.log_softmax(tl(x.to("mps"))[0, -1].float(), -1).cpu() for x in ids]
        del tl
        gc.collect()
        torch.mps.empty_cache()

    def diff(a, b):
        full = [(x - y).abs().max().item() for x, y in zip(lp[a], lp[b])]
        top10 = [(x - y).abs()[y.topk(10).indices].max().item() for x, y in zip(lp[a], lp[b])]
        return {"max_abs_diff": max(full), "max_abs_diff_top10_of_second": max(top10),
                "per_prompt_max": full}
    pairs = {"hf_fp32_vs_tl_fp32": (("hf", "float32"), ("tl", "float32")),
             "hf_bf16_vs_tl_bf16": (("hf", "bfloat16"), ("tl", "bfloat16")),
             "hf_fp32_vs_hf_bf16": (("hf", "float32"), ("hf", "bfloat16")),
             "tl_fp32_vs_tl_bf16": (("tl", "float32"), ("tl", "bfloat16"))}
    m = {"model": name, "n": args.n, "prompt_seeds": seeds, "seed": args.seed,
         "comparisons": {k: diff(*v) for k, v in pairs.items()}}
    m["top1_agree_hf_fp32_vs_hf_bf16"] = sum(int(x.argmax() == y.argmax()) for x, y in
                                            zip(lp["hf", "float32"], lp["hf", "bfloat16"]))
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_prompt_max"}
                      for k, v in m["comparisons"].items()}, indent=1))


if __name__ == "__main__":
    main()
