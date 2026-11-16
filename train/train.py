"""Training loop.

AdamW, 1% linear warmup then cosine decay to min_lr, grad clip, float32 on MPS by default.
Writes results/<task>/<run_name>/{metrics.json, cmd.txt, log.jsonl}
and checkpoints/<task>/<run_name>/step_<n>.pt at the configured fractions of total steps.

Data sources:
  --data random            uniform random tokens (smoke test)
  --data <path.bin>        uint16 token stream, read sequentially in a fixed order
"""
import argparse
import json
import math
import os
import shlex
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.gpt import GPT, GPTConfig  # noqa: E402


def lr_at(step, total, cfg):
    warmup = max(1, int(round(cfg["warmup_frac"] * total)))
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return cfg["min_lr"] + 0.5 * (cfg["lr"] - cfg["min_lr"]) * (1 + math.cos(math.pi * progress))


class RandomTokens:
    def __init__(self, vocab_size, batch_size, seq_len, seed):
        self.g = torch.Generator().manual_seed(seed)
        self.v, self.b, self.t = vocab_size, batch_size, seq_len

    def next(self):
        x = torch.randint(0, self.v, (self.b, self.t + 1), generator=self.g)
        return x[:, :-1], x[:, 1:]


class TokenStream:
    """Sequential, fixed-order reader over a uint16 token file (wraps around at the end)."""

    def __init__(self, path, batch_size, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.b, self.t = batch_size, seq_len
        self.pos = 0

    def next(self):
        n = self.b * (self.t + 1)
        if self.pos + n > len(self.data):
            self.pos = 0
        chunk = np.asarray(self.data[self.pos:self.pos + n], dtype=np.int64)
        self.pos += n
        x = torch.from_numpy(chunk).view(self.b, self.t + 1)
        return x[:, :-1], x[:, 1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model_config", default="configs/model_small.json")
    p.add_argument("--train_config", default="configs/train.json")
    p.add_argument("--data", default="random")
    p.add_argument("--steps", type=int, required=True, help="total steps (sets the LR schedule)")
    p.add_argument("--stop_after", type=int, default=None, help="stop early after this many steps")
    p.add_argument("--amp", choices=["none", "fp16"], default="none")
    p.add_argument("--device", default="mps")
    p.add_argument("--no_ckpt", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    os.chdir(ROOT)
    out_dir = os.path.join("results", args.task, args.run_name)
    ckpt_dir = os.path.join("checkpoints", args.task, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cmd.txt"), "w") as f:
        f.write("PYTORCH_ENABLE_MPS_FALLBACK=1 " + shlex.join([sys.executable] + sys.argv) + "\n")

    with open(args.model_config) as f:
        mcfg = GPTConfig(**json.load(f))
    with open(args.train_config) as f:
        tcfg = json.load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model = GPT(mcfg).to(device)
    opt = model.configure_optimizer(tcfg["lr"], tuple(tcfg["betas"]), tcfg["weight_decay"])

    B, T = tcfg["batch_size"], tcfg["seq_len"]
    if args.data == "random":
        loader = RandomTokens(mcfg.vocab_size, B, T, args.seed)
    else:
        loader = TokenStream(args.data, B, T)

    total = args.steps
    run_steps = min(total, args.stop_after) if args.stop_after else total
    ckpt_steps = {max(1, int(round(fr * total))) for fr in tcfg["checkpoint_fracs"]}

    log_f = open(os.path.join(out_dir, "log.jsonl"), "w")
    losses, step_times = [], []
    t_start = time.time()
    for step in range(run_steps):
        lr = lr_at(step, total, tcfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = loader.next()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        t0 = time.time()
        if args.amp == "fp16":
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
        opt.step()
        loss_v = loss.item()  # synchronises the device
        dt = time.time() - t0

        if not math.isfinite(loss_v):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_v}")
        losses.append(loss_v)
        step_times.append(dt)
        rec = {"step": step + 1, "loss": loss_v, "lr": lr, "grad_norm": float(gnorm), "dt": dt}
        log_f.write(json.dumps(rec) + "\n")
        if (step + 1) % args.log_every == 0 or step == 0:
            log_f.flush()
            print(f"step {step+1}/{run_steps} loss {loss_v:.4f} lr {lr:.2e} "
                  f"tok/s {B*T/dt:,.0f}", flush=True)

        if not args.no_ckpt and (step + 1) in ckpt_steps:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                        "total_steps": total, "model_config": vars(mcfg), "train_config": tcfg,
                        "args": vars(args)},
                       os.path.join(ckpt_dir, f"step_{step+1}.pt"))
    log_f.close()

    skip = min(5, len(step_times) // 2)  # ignore warm-up/compilation steps for throughput
    steady = step_times[skip:]
    tok_per_s = B * T * len(steady) / sum(steady)
    metrics = {
        "steps_run": run_steps, "total_steps": total, "seed": args.seed, "amp": args.amp,
        "data": args.data, "tokens_per_step": B * T,
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_first10_mean": float(np.mean(losses[:10])),
        "loss_last10_mean": float(np.mean(losses[-10:])),
        "tokens_per_second": tok_per_s,
        "wall_seconds": time.time() - t_start,
        "non_embedding_params": model.n_params(True),
        "total_params": model.n_params(False),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
