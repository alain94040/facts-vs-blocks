"""Hop task 0: set-up and checks (hop/PLAN.md). Gate H0.

1. downloads (hop/download.py) and a check that the transcoders circuit-tracer loads for "gemma"
   (mwhanna/gemma-scope-transcoders) equal the GemmaScope originals named in
   mntss/gemma-scope-transcoders (max absolute difference per parameter, all 26 layers);
2. next-token log-probs of the plain TransformerLens model (loaded with the same options as the
   ReplacementModel: fold_ln, center_writing_weights, center_unembed all False) vs the
   ReplacementModel, on the TWO prompts of TARGET and 5 CAP items (seed 0); max |difference|;
3. identity check: feature_intervention setting the 20 most active features at the last position
   to their own values; max |change| in log-probs;
4. the TARGET graph: attribute(max_feature_nodes=4096, batch_size=64); time, peak memory (RSS +
   MPS driver), active features, replacement and completeness scores. The graph is saved to
   data/cache/hop/graphs/TARGET.pt for Task 2.
Usage: python hop/task0.py --seed 0
Writes results/hop/task0/{metrics.json, cmd.txt}.
"""
import argparse
import inspect
import json
import os
import random
import time

import common
import data
import numpy as np
import torch
from safetensors.numpy import load_file


def check_transcoders():
    from huggingface_hub import snapshot_download
    import yaml
    from huggingface_hub import hf_hub_download
    sd = snapshot_download("mwhanna/gemma-scope-transcoders", allow_patterns=["layer_*.safetensors"])
    cfg = yaml.safe_load(open(hf_hub_download("mntss/gemma-scope-transcoders", "config.yaml")))
    g = "google/gemma-scope-2b-pt-transcoders"
    res = {"n_layers": 0, "keys_safetensors": None, "keys_npz": None, "max_abs_diff": {}}
    worst = 0.0
    for layer, uri in enumerate(cfg["transcoders"]):
        a = load_file(os.path.join(sd, f"layer_{layer}.safetensors"))
        b = dict(np.load(hf_hub_download(g, uri.split(g + "/", 1)[1])))
        if res["keys_safetensors"] is None:
            res["keys_safetensors"] = {k: list(v.shape) for k, v in a.items()}
            res["keys_npz"] = {k: list(v.shape) for k, v in b.items()}
        pairs = {"W_enc": ("W_enc", "W_enc"), "W_dec": ("W_dec", "W_dec"), "b_enc": ("b_enc", "b_enc"),
                 "b_dec": ("b_dec", "b_dec"),
                 "threshold": (next(k for k in a if "threshold" in k), "threshold")}
        for name, (ka, kb) in pairs.items():
            x, y = a[ka].astype(np.float64), b[kb].astype(np.float64)
            if x.shape != y.shape:
                x = x.T
            d = float(np.abs(x - y).max())
            res["max_abs_diff"][f"{layer}.{name}"] = d
            worst = max(worst, d)
        res["n_layers"] += 1
    res["max_abs_diff_all"] = worst
    res["same"] = worst == 0.0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    d = common.out_dir("task0")
    m = {"seed": args.seed, "model": common.MODEL, "transcoder_set": common.TRANSCODERS,
         "device": str(common.DEVICE), "dtype": "float32"}
    dl = json.load(open(os.path.join(d, "download", "downloads.json")))
    m["downloads"] = dl
    import circuit_tracer
    import transformer_lens
    import transformers
    import safetensors
    m["versions"] = {"circuit_tracer": circuit_tracer.__version__, "torch": torch.__version__,
                     "transformer_lens": transformer_lens.__version__ if hasattr(transformer_lens, "__version__") else None,
                     "transformers": transformers.__version__, "safetensors": safetensors.__version__}

    print("step 1b: transcoder equality", flush=True)
    m["transcoder_check"] = check_transcoders()
    print({k: v for k, v in m["transcoder_check"].items() if k != "max_abs_diff"}, flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(common.MODEL)
    its = data.items(tok)
    cap = [it for it in its if it["set"] == "CAP"]
    chosen = [its[0]] + random.Random(args.seed).sample(cap, 5)
    assert chosen[0]["id"] == "TARGET"
    m["prompts"] = [{"id": it["id"], "prompt": it["TWO"]} for it in chosen]
    ids = [torch.tensor([common.tokenize(tok, it["TWO"])], device=common.DEVICE) for it in chosen]

    mem = common.MemSampler()
    mem.start()

    print("step 2: plain TransformerLens", flush=True)
    from transformer_lens import HookedTransformer
    tl = HookedTransformer.from_pretrained(common.MODEL, fold_ln=False, center_writing_weights=False,
                                           center_unembed=False, device=common.DEVICE, dtype=common.DTYPE)
    tl.eval()
    with torch.no_grad():
        lp_tl = [torch.log_softmax(tl(x)[0, -1].cpu().double(), -1) for x in ids]
    del tl
    common.free()

    t0 = time.time()
    rm = common.load_replacement_model()
    m["load_seconds"] = time.time() - t0
    m["memory_after_load_gb"] = mem.gb()
    with torch.no_grad():
        lp_rm = [torch.log_softmax(rm(x)[0, -1].cpu().double(), -1) for x in ids]
    per = []
    for it, a, b in zip(chosen, lp_tl, lp_rm):
        per.append({"id": it["id"], "max_abs_diff_logprob": float((a - b).abs().max()),
                    "tl_top": tok.decode([int(a.argmax())]), "rm_top": tok.decode([int(b.argmax())])})
    m["step2_logit_check"] = {"per_prompt": per, "max_abs_diff": max(p["max_abs_diff_logprob"] for p in per)}
    print(m["step2_logit_check"], flush=True)

    print("step 3: identity intervention", flush=True)
    sig = inspect.signature(rm.feature_intervention)
    m["feature_intervention"] = {
        "signature": str(sig),
        "defaults_used": {k: repr(v.default) for k, v in sig.parameters.items()
                          if v.default is not inspect.Parameter.empty}}
    per3 = []
    for it, x, b in zip(chosen, ids, lp_rm):
        with torch.no_grad():
            _, acts = rm.get_activations(x)
        last = acts[:, -1].float()  # [layers, features]
        v, flat = last.flatten().topk(20)
        n_feat = last.shape[1]
        iv = [(int(f) // n_feat, x.shape[1] - 1, int(f) % n_feat, float(val)) for f, val in zip(flat, v)]
        with torch.no_grad():
            logits, _ = rm.feature_intervention(x, iv)
        lp = torch.log_softmax(logits[0, -1].cpu().double(), -1)
        per3.append({"id": it["id"], "n_features": len(iv), "min_value": float(v.min()),
                     "max_abs_change_logprob": float((lp - b).abs().max())})
        del acts
        common.free()
    m["step3_identity"] = {"per_prompt": per3, "max_abs_change": max(p["max_abs_change_logprob"] for p in per3)}
    print(m["step3_identity"], flush=True)

    print("step 4: TARGET graph", time.strftime("%H:%M:%S"), flush=True)
    from circuit_tracer import attribute
    from circuit_tracer.graph import compute_graph_scores
    common.free()
    peak_before_graph = mem.gb()
    mem.reset()
    torch.mps.synchronize()
    t0 = time.time()
    g = attribute(ids[0][0], rm, max_feature_nodes=4096, batch_size=64, verbose=True)
    torch.mps.synchronize()
    secs = time.time() - t0
    peak_graph = mem.gb()
    rep, comp = compute_graph_scores(g)
    os.makedirs(os.path.join(common.CACHE, "graphs"), exist_ok=True)
    g.to_pt(os.path.join(common.CACHE, "graphs", "TARGET.pt"))
    m["step4_graph"] = {"prompt": chosen[0]["TWO"], "n_tokens": int(ids[0].shape[1]),
                        "attribute_args": {"max_feature_nodes": 4096, "batch_size": 64},
                        "seconds": secs, "peak_memory_gb_during_graph": peak_graph,
                        "n_active_features": int(g.active_features.shape[0]),
                        "n_selected_features": int(len(g.selected_features)),
                        "replacement_score": rep, "completeness_score": comp,
                        "logit_targets": [tok.decode([t.vocab_idx]) for t in g.logit_targets],
                        "logit_probabilities": [float(p) for p in g.logit_probabilities]}
    del g
    common.free()
    mem.stop = True
    mem.sample()
    m["peak_memory_gb_before_graph"] = peak_before_graph  # includes the plain TransformerLens model of step 2
    m["peak_memory_gb_whole_run"] = {k: max(peak_before_graph[k], mem.gb()[k]) for k in peak_before_graph}
    m["gate_H0"] = {"step2_le_0.01": m["step2_logit_check"]["max_abs_diff"] <= 0.01,
                    "step3_le_0.001": m["step3_identity"]["max_abs_change"] <= 0.001,
                    "graph_le_20min": secs <= 20 * 60,
                    "peak_le_48GB": peak_graph["sum"] <= 48}
    m["gate_H0"]["pass"] = all(m["gate_H0"].values())
    common.dump(m, os.path.join(d, "metrics.json"))
    print(json.dumps({k: m[k] for k in ("step2_logit_check", "step3_identity", "step4_graph",
                                          "peak_memory_gb_whole_run", "gate_H0")}, indent=1))


if __name__ == "__main__":
    main()
