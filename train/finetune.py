"""Task 4b: fine-tune one part of a model on new people, everything else frozen (PLAN.md Phase 2b).

Variants (trainable parameters):
  F-all   all weights (including LayerNorms and position embeddings)
  F-attn  attention qkv and out (weights and biases) of every block
  F-mlp   MLP up and down (weights and biases) of every block
  F-mlp34 MLP up and down of layers 3 and 4
  F-L0    attention and MLP of layer 0
  F-emb   token embeddings (tied with the output layer)
  none    no training: measures the starting model
LayerNorms and position embeddings are frozen in every variant except F-all.

AdamW (configs/finetune.json): weight decay on trainable matrices (dim >= 2), none on trainable
biases/LayerNorm gains, as in Phase 1. 1% warmup, cosine to min_lr, clip 1.0 over trainable
parameters, one pass over the stream, float32.

Measures at the end: recall on set-B people 0..999 (train_templates, template5_seen,
template5_heldout), recall on set-A people 0..999, STORY loss, FORMAT loss, name-copy accuracy,
number of trainable parameters.

  python train/finetune.py --init fork-A-s0 --variant F-mlp --stream ft-B1k --seed 0
"""
import argparse
import json
import os
import shlex
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from analysis import common as C  # noqa: E402
from train import evaluate as ev  # noqa: E402
from train.train import TokenStream, lr_at  # noqa: E402

VARIANTS = ["none", "F-all", "F-attn", "F-mlp", "F-mlp34", "F-L0", "F-emb"]
N_EVAL_PEOPLE = 1000


def trainable(name, variant):
    if variant == "F-all":
        return True
    if variant == "none":
        return False
    parts = name.split(".")
    if variant == "F-emb":
        return name in ("tok_emb.weight", "head.weight")
    if parts[0] != "blocks":
        return False
    layer, mod = int(parts[1]), parts[2]
    if variant == "F-attn":
        return mod == "attn"
    if variant == "F-mlp":
        return mod == "mlp"
    if variant == "F-mlp34":
        return mod == "mlp" and layer in (3, 4)
    if variant == "F-L0":
        return mod in ("attn", "mlp") and layer == 0
    raise ValueError(variant)


def measure(model, start_meta):
    out = {}
    for s in "AB":
        with open(os.path.join(ROOT, "data", "cache", "bios", f"people_{s}.json")) as f:
            ppl = json.load(f)[:N_EVAL_PEOPLE]
        r = ev.fact_eval(model, ev.build_prompts(list(enumerate(ppl))), C.DEVICE)
        out[f"set_{s}_0_999"] = r
    model.eval()
    out["story_loss"] = C.mean_loss(model, "STORY", start_meta)
    out["format_loss"] = C.mean_loss(model, "FORMAT", start_meta)
    out["name_copy"] = ev.name_copy(model, C.DEVICE, start_meta)["accuracy"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="fork-A-s0", help="run whose final checkpoint to start from")
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--stream", default="ft-B1k")
    p.add_argument("--config", default="configs/finetune.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", default="task4b")
    p.add_argument("--run_name", default=None)
    p.add_argument("--stop_after", type=int, default=None)
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--log_every", type=int, default=25)
    args = p.parse_args()
    os.chdir(ROOT)
    run_name = args.run_name or args.variant
    out_dir = os.path.join("results", args.task, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")
    with open(args.config) as f:
        cfg = json.load(f)
    torch.manual_seed(args.seed)

    model = C.load_model(args.init)
    start_meta = C.run_meta(args.init)
    named = dict(model.named_parameters())  # tied head.weight appears once, as tok_emb.weight
    train_p = {n: q for n, q in named.items() if trainable(n, args.variant)}
    for n, q in named.items():
        q.requires_grad_(n in train_p)
    n_train = sum(q.numel() for q in train_p.values())
    metrics = {"variant": args.variant, "init": args.init, "init_ckpt": C.ckpt_path(args.init),
               "seed": args.seed, "config": cfg, "trainable_params": n_train,
               "trainable_tensors": sorted(train_p)}
    print(args.variant, "trainable params", n_train, flush=True)

    if train_p:
        decay = [q for q in train_p.values() if q.dim() >= 2]
        no_decay = [q for q in train_p.values() if q.dim() < 2]
        groups = [g for g in [{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}] if g["params"]]
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=tuple(cfg["betas"]))
        with open(os.path.join("data", "cache", "streams", f"{args.stream}.json")) as f:
            smeta = json.load(f)
        B, T = cfg["batch_size"], cfg["seq_len"]
        assert smeta["tokens_per_step"] == B * (T + 1)
        total = smeta["steps"]
        run_steps = min(total, args.stop_after) if args.stop_after else total
        loader = TokenStream(os.path.join("data", "cache", "streams", f"{args.stream}.bin"), B, T)
        model.train()
        losses, times = [], []
        log_f = open(os.path.join(out_dir, "log.jsonl"), "w")
        t_start = time.time()
        for step in range(run_steps):
            lr = lr_at(step, total, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = loader.next()
            t0 = time.time()
            _, loss = model(x.to(C.DEVICE), y.to(C.DEVICE))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(list(train_p.values()), cfg["grad_clip"])
            opt.step()
            lv = loss.item()
            times.append(time.time() - t0)
            losses.append(lv)
            if not np.isfinite(lv):
                raise RuntimeError(f"non-finite loss at step {step}")
            log_f.write(json.dumps({"step": step + 1, "loss": lv, "lr": lr,
                                    "grad_norm": float(gn)}) + "\n")
            if (step + 1) % args.log_every == 0 or step == 0:
                log_f.flush()
                print(f"step {step+1}/{run_steps} loss {lv:.4f} lr {lr:.2e} "
                      f"tok/s {B*T/times[-1]:,.0f}", flush=True)
        log_f.close()
        steady = times[min(5, len(times) // 2):]
        metrics.update({"stream": args.stream, "total_steps": total, "steps_run": run_steps,
                        "loss_first10_mean": float(np.mean(losses[:10])),
                        "loss_last10_mean": float(np.mean(losses[-10:])),
                        "tokens_per_second": B * T * len(steady) / sum(steady),
                        "train_seconds": time.time() - t_start})
        for q in named.values():
            q.requires_grad_(True)
    if not args.no_eval:
        metrics["eval"] = measure(model.eval(), start_meta)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "trainable_tensors"}, indent=1)[:3000])


if __name__ == "__main__":
    main()
