"""Circuit task 0: do circuit-tracer and the transcoders run on the Mac? (circuit/PLAN.md)

Usage: python circuit/task0.py --which dev --n 20 --seed 0
       python circuit/task0.py --which main --n 5 --seed 0
Writes results/circuit/task0/<which>/{metrics.json, cmd.txt}.

Prompts are PROG P1 items with seeds 9001.. (outside every seed range the plan uses later).
1. plain transformers: final-position log-probabilities (bfloat16, MPS);
2. circuit_tracer.ReplacementModel (transformerlens backend, library defaults): same log-probs;
   max |difference| over the full vocabulary, per prompt;
3. one attribution graph per prompt with attribute()'s defaults: wall time, replacement and
   completeness scores (compute_graph_scores on the full graph), node counts before/after the
   library's default pruning.
Memory is sampled every 0.5 s: process RSS, MPS driver-allocated memory, and their sum.
"""
import argparse
import gc
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import psutil  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

import tasks  # noqa: E402

MODELS = {"dev": ("Qwen/Qwen3-0.6B", "mwhanna/qwen3-0.6b-transcoders-lowl0"),
          "main": ("Qwen/Qwen3-4B", "mwhanna/qwen3-4b-transcoders")}


class MemSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.proc = psutil.Process()
        self.peak = {"rss": 0, "mps_driver": 0, "sum": 0}
        self.stop = False

    def sample(self):
        rss = self.proc.memory_info().rss
        drv = torch.mps.driver_allocated_memory() if torch.backends.mps.is_available() else 0
        for k, v in (("rss", rss), ("mps_driver", drv), ("sum", rss + drv)):
            self.peak[k] = max(self.peak[k], v)

    def run(self):
        while not self.stop:
            self.sample()
            time.sleep(0.5)

    def gb(self):
        return {k: round(v / 2**30, 2) for k, v in self.peak.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=list(MODELS), required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lazy-encoder", action="store_true", help="library option: read encoders from disk")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="attribute() batch_size; omitted = library default (512)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out_dir = os.path.join(ROOT, "results", "circuit", "task0", args.which + args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    model_name, tc_name = MODELS[args.which]
    dtype = getattr(torch, args.dtype)
    dev = torch.device(args.device)
    api = HfApi()
    m = {"model": model_name, "model_revision": api.model_info(model_name).sha,
         "transcoders": tc_name, "transcoder_revision": api.model_info(tc_name).sha,
         "device": args.device, "dtype": args.dtype, "seed": args.seed, "n": args.n,
         "prompt_seeds": list(range(9001, 9001 + args.n)), "family": "PROG", "level": "P1"}
    import circuit_tracer
    m["circuit_tracer_version"] = circuit_tracer.__version__
    items = [tasks.make("PROG", "P1", s) for s in m["prompt_seeds"]]

    mem = MemSampler()
    mem.start()

    # 1. plain transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(dev).eval()
    ref = []
    with torch.no_grad():
        for it in items:
            ids = tok(it["prompt"], return_tensors="pt")["input_ids"].to(dev)
            ref.append((ids.cpu(), torch.log_softmax(hf(ids).logits[0, -1].float(), -1).cpu()))
    del hf
    gc.collect()
    torch.mps.empty_cache()

    # 2. replacement model
    import ct_compat  # noqa: F401  (mps:0 -> mps for safetensors)
    from circuit_tracer import ReplacementModel, attribute
    from circuit_tracer.graph import compute_graph_scores, prune_graph
    t0 = time.time()
    try:
        rm = ReplacementModel.from_pretrained(model_name, tc_name, device=dev, dtype=dtype,
                                             lazy_encoder=args.lazy_encoder, lazy_decoder=True)
        m["replacement_model_device"] = args.device
    except Exception as e:  # noqa: BLE001
        m["mps_error"] = repr(e)[:2000]
        print("MPS load failed:", e, "-> CPU", flush=True)
        dev = torch.device("cpu")
        rm = ReplacementModel.from_pretrained(model_name, tc_name, device=dev, dtype=dtype,
                                             lazy_encoder=args.lazy_encoder, lazy_decoder=True)
        m["replacement_model_device"] = "cpu"
    m["load_seconds"] = time.time() - t0
    m["attribute_batch_size"] = args.batch_size or "library default (512)"
    m["lazy_encoder"], m["lazy_decoder"] = args.lazy_encoder, True  # decoder lazy = library default
    m["memory_after_load_gb"] = mem.gb()

    per = []
    with torch.no_grad():
        for (ids, lp_ref), it in zip(ref, items):
            ids_rm = rm.ensure_tokenized(it["prompt"])
            same_tokens = ids_rm.cpu().flatten().tolist() == ids.flatten().tolist()
            lp = torch.log_softmax(rm(ids.to(dev))[0, -1].float(), -1).cpu()
            d = (lp - lp_ref).abs()
            gold = tok.convert_tokens_to_ids(it["answer"])
            per.append({"seed": it["seed"], "n_tokens": ids.shape[1], "same_tokens": same_tokens,
                        "max_abs_diff_logprob": d.max().item(),
                        "max_abs_diff_top10": d[lp_ref.topk(10).indices].max().item(),
                        "gold_logprob_hf": lp_ref[gold].item(), "gold_logprob_rm": lp[gold].item(),
                        "hf_top": tok.decode([lp_ref.argmax().item()]),
                        "rm_top": tok.decode([lp.argmax().item()])})
    m["logit_check"] = {"max_abs_diff": max(p["max_abs_diff_logprob"] for p in per),
                        "max_abs_diff_top10": max(p["max_abs_diff_top10"] for p in per),
                        "pass": max(p["max_abs_diff_logprob"] for p in per) <= 0.05}
    print(json.dumps(m["logit_check"]), flush=True)

    # 3. attribution graphs, library defaults
    for p, it in zip(per, items):
        if dev.type == "mps":
            torch.mps.synchronize()
        print("graph for seed", it["seed"], "started", time.strftime("%H:%M:%S"), flush=True)
        t0 = time.time()
        kw = {} if args.batch_size is None else {"batch_size": args.batch_size}
        g = attribute(it["prompt"], rm, verbose=True, **kw)
        if dev.type == "mps":
            torch.mps.synchronize()
        p["graph_seconds"] = time.time() - t0
        rep, comp = compute_graph_scores(g)
        pr = prune_graph(g)
        n_feat = len(g.selected_features)
        p.update({"replacement_score": rep, "completeness_score": comp,
                  "n_active_features": int(g.active_features.shape[0]), "n_selected_features": n_feat,
                  "n_nodes": int(g.adjacency_matrix.shape[0]),
                  "n_nodes_pruned": int(pr.node_mask.sum()), "n_edges_pruned": int(pr.edge_mask.sum()),
                  "n_feature_nodes_pruned": int(pr.node_mask[:n_feat].sum()),
                  "logit_targets": len(g.logit_targets)})
        print({k: p[k] for k in ("seed", "graph_seconds", "replacement_score", "completeness_score",
                                  "n_nodes_pruned")}, mem.gb(), flush=True)
        del g, pr
        gc.collect()
    mem.stop = True
    mem.sample()
    m["per_prompt"] = per
    secs = [p["graph_seconds"] for p in per]
    m["graph_seconds_mean"] = sum(secs) / len(secs)
    m["graph_seconds_max"] = max(secs)
    m["peak_memory_gb"] = mem.gb()
    for k in ("replacement_score", "completeness_score"):
        v = sorted(p[k] for p in per)
        m[k + "_median"] = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
    m["gate_G0"] = {"logits_match": m["logit_check"]["pass"],
                    "graph_le_20min": m["graph_seconds_max"] <= 20 * 60,
                    "peak_le_48GB": m["peak_memory_gb"]["sum"] <= 48}
    m["gate_G0"]["pass"] = all(m["gate_G0"].values())
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(json.dumps({k: m[k] for k in ("logit_check", "graph_seconds_mean", "graph_seconds_max",
                                          "peak_memory_gb", "replacement_score_median",
                                          "completeness_score_median", "gate_G0")}, indent=1))


if __name__ == "__main__":
    main()
