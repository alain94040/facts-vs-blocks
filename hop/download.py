"""Hop task 0, step 1: download the model and the transcoders into data/cache/hf/ and record sizes
and revisions (results/hop/task0/download/downloads.json).

Downloads:
  - google/gemma-2-2b: config, tokenizer, safetensors weights;
  - mwhanna/gemma-scope-transcoders: what circuit-tracer 0.5.0 loads for the transcoder set "gemma"
    (layer_*.safetensors, config.yaml);
  - the GemmaScope originals listed in mntss/gemma-scope-transcoders/config.yaml (params.npz from
    google/gemma-scope-2b-pt-transcoders), to check that the two repos hold the same transcoders
    (hop/check_transcoders.py).
Usage: python hop/download.py --seed 0
"""
import argparse
import shutil

import common
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()
    d = common.out_dir("task0", "download")
    api = HfApi()
    rec = {"disk_free_gb_before": round(shutil.disk_usage(common.ROOT).free / 1e9, 1), "repos": {}}
    print(rec, flush=True)

    def info(repo, files):
        i = api.model_info(repo, files_metadata=True)
        sizes = {s.rfilename: s.size for s in i.siblings if s.rfilename in files}
        return {"revision": i.sha, "n_files": len(sizes), "bytes": sum(sizes.values()),
                "gb": round(sum(sizes.values()) / 1e9, 3)}

    snapshot_download(common.MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
    files = [s.rfilename for s in api.model_info(common.MODEL).siblings
             if s.rfilename.endswith((".json", ".safetensors")) or s.rfilename.startswith("tokenizer")]
    rec["repos"][common.MODEL] = info(common.MODEL, files)
    print(rec, flush=True)

    tc = "mwhanna/gemma-scope-transcoders"
    snapshot_download(tc, allow_patterns=["layer_*.safetensors", "config.yaml"])
    files = [s.rfilename for s in api.model_info(tc).siblings
             if s.rfilename.startswith("layer_") or s.rfilename == "config.yaml"]
    rec["repos"][tc] = info(tc, files)
    print(rec, flush=True)

    import yaml
    cfg = yaml.safe_load(open(hf_hub_download("mntss/gemma-scope-transcoders", "config.yaml")))
    rec["repos"]["mntss/gemma-scope-transcoders"] = {
        "revision": api.model_info("mntss/gemma-scope-transcoders").sha, "transcoders": cfg["transcoders"]}
    g = "google/gemma-scope-2b-pt-transcoders"
    files = [p.split(g + "/", 1)[1] for p in cfg["transcoders"]]
    for f in files:
        hf_hub_download(g, f)
    rec["repos"][g] = info(g, files)
    rec["disk_free_gb_after"] = round(shutil.disk_usage(common.ROOT).free / 1e9, 1)
    common.dump(rec, f"{d}/downloads.json")
    print(rec, flush=True)


if __name__ == "__main__":
    main()
