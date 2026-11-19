"""Circuit task 0: gather every Task 0 measurement into results/circuit/task0/metrics.json.

Usage: python circuit/task0_summary.py --seed 0   (no model; reads the per-run metrics.json files)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "circuit", "task0")


def load(name):
    return json.load(open(os.path.join(D, name, "metrics.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    downloads = [json.loads(line) for line in open(os.path.join(D, "downloads.jsonl"))]
    dev = load("dev_float32")
    dt = {w: load(f"dtype_{w}") for w in ("dev", "main")}
    n_feat, n_pos, n_layers = 90240, 83, 36
    m = {
        "seed": args.seed,
        "downloads": [{"repo": d["repo"], "revision": d["revision"], "gb": round(d["bytes"] / 1e9, 2),
                       "free_disk_before_gb": round(d["free_before"] / 1e9, 1),
                       "free_disk_after_gb": round(d["free_after"] / 1e9, 1)} for d in downloads],
        "software": {"circuit_tracer": dev["circuit_tracer_version"], "backend": "transformerlens",
                     "safetensors": "0.7.0 (0.8.0 rejects the tensor index used by the lazy decoder)",
                     "compat_patch": "circuit/ct_compat.py: safe_open device 'mps:0' -> 'mps' (library not edited)"},
        "decisions_approved_by_reviewer_2026-11-18": {
            "dtype": "float32 for the replacement model (the 0.05 logit limit applies within one dtype; "
                     "bfloat16 failing it is not a gate failure)",
            "lazy_encoder": "main model transcoders: float32 encoders would take ~60 GB, so encoders are "
                            "read from disk (lazy_encoder=True); decoders lazy (library default)",
            "safetensors_and_patch": "safetensors 0.7.0 and circuit/ct_compat.py"},
        "dtype_check": {w: {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt_max"}
                            for k, v in dt[w]["comparisons"].items()} | {"model": dt[w]["model"], "n": dt[w]["n"]}
                        for w in dt},
        "dev": {k: dev[k] for k in ("model", "model_revision", "transcoders", "transcoder_revision",
                                    "device", "dtype", "lazy_encoder", "attribute_batch_size" if "attribute_batch_size" in dev else "lazy_decoder",
                                    "n", "logit_check", "graph_seconds_mean", "graph_seconds_max",
                                    "peak_memory_gb", "replacement_score_median", "completeness_score_median")},
        "main_runs": {r: load(r) for r in ("main_float32_lazyenc", "main_float32_lazyenc_cpu",
                                           "main_float32_lazyenc_bs128_INFO")},
        "main_edge_matrix_gb_computed": {
            "note": "attribute() allocates edge_matrix = (n_feature_nodes + n_logits) x (n_features + (n_layers+1)*n_pos + n_logits) "
                    "float32 on CPU; with the default max_feature_nodes=None every active feature is a node",
            "n_active_features": n_feat, "n_pos": n_pos, "n_layers": n_layers,
            "gb": round((n_feat + 1) * (n_feat + (n_layers + 1) * n_pos + 1) * 4 / 1e9, 1)},
    }
    m["dev"]["model_revision"] = dev["model_revision"]
    m["revision2_defaults_gate_G0"] = {
        "pass": False,
        "why": "library defaults: MPS crashes (MPSGraph INT_MAX); CPU needs ~57 min per graph; "
               "edge matrix alone ~34 GB plus 16 GB float32 model weights."}

    # revision 3: float32, lazy encoder (main), batch_size 64, max_feature_nodes 8192 then 4096
    keys = ("logit_check", "memory_after_load_gb", "graph_seconds_mean", "graph_seconds_max",
            "peak_memory_gb", "replacement_score_median", "completeness_score_median",
            "swap_used_gb_at_start", "attribute_batch_size", "max_feature_nodes", "lazy_encoder")
    r3 = {"settings": {"dtype": "float32", "lazy_encoder_main": True, "safetensors": "0.7.0",
                       "compat_patch": "circuit/ct_compat.py", "batch_size": 64,
                       "max_feature_nodes": "8192, then 4096 if G0 fails"}}
    for name in ("dev_r3_cap8192", "dev_r3_nocap", "main_r3_cap4096"):
        x = load(name)
        r3[name] = {k: x.get(k) for k in keys}
        r3[name]["per_prompt"] = [{k: p[k] for k in ("seed", "graph_seconds", "replacement_score",
                                                     "completeness_score", "n_active_features",
                                                     "n_nodes_pruned", "n_feature_nodes_pruned")}
                                  for p in x["per_prompt"]]
    r3["main_r3_cap8192"] = load("main_r3_cap8192")
    r3["cap_effect_dev"] = load("cap_effect")["median"]
    c4 = load("main_r3_cap4096")
    r3["gate_G0"] = {
        "at_8192": {"logits_match": True, "graph_le_20min": True, "peak_le_48GB": False, "pass": False,
                    "note": "one graph built (12.6 min); peak sum 90.9 GB; run stopped after it"},
        "at_4096": c4["gate_G0"] | {"peak_sum_gb": c4["peak_memory_gb"]["sum"],
                                    "graph_seconds_max": c4["graph_seconds_max"]},
        "pass": False,
        "memory_measure": "peak of (process RSS + MPS driver-allocated memory), sampled every 0.5 s; "
                          "driver memory includes the MPS allocator's cache",
    }
    m["revision3"] = r3
    m["gate_G0"] = r3["gate_G0"]
    json.dump(m, open(os.path.join(D, "metrics.json"), "w"), indent=1)
    with open(os.path.join(D, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    print(json.dumps(m["gate_G0"], indent=1))


if __name__ == "__main__":
    main()
