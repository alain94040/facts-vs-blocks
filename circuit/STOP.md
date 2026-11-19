# STOP: gate G0 fails on the main model (circuit task 0)

Gate G0 (circuit/PLAN.md), main model `Qwen/Qwen3-4B` with `mwhanna/qwen3-4b-transcoders`, 5 PROG P1
prompts (seeds 9001–9005): logits match within 0.05, **one graph ≤ 20 minutes**, **peak memory ≤ 48 GB**.
The logit check passes; graph time and memory fail. Task 1 was not started. All numbers are in
`results/circuit/task0/metrics.json`.

Settings approved by the reviewer on 2026-11-18: float32, lazy encoder (`lazy_encoder=True`;
float32 encoders would take ~60 GB), safetensors 0.7.0 and `circuit/ct_compat.py`. Graphs use
`attribute(prompt, model)` with library defaults (circuit-tracer 0.5.0, transformerlens backend;
`batch_size=512`, `max_feature_nodes=None`, `desired_logit_prob=0.95`, `max_n_logits=10`).

## Main model

| Run | Device | Logits (max abs diff) | Graph | Result |
| --- | --- | --- | --- | --- |
| defaults | MPS | 1.8e-4 (pass) | crashed in the feature phase: `MPSGraph does not support tensor dims larger than INT_MAX` | no graph |
| defaults (the plan's CPU fallback) | CPU | 1.8e-4 (pass) | 90,241 active features; forward 231 s, logit phase 77 s; feature phase 37,888/90,241 after 15 min, library ETA 37 min more: **~57 min per graph** | stopped at 20.7 min |
| `batch_size=128` (**informational, not a G0 setting**) | MPS | 1.8e-4 (pass) | 90,240 active features; feature phase at 22 it/s, then 5 it/s once the system swapped (33.5 GB swap); 56,576/90,240 after 28 min, library ETA 1 h 47 min more | stopped at 29.1 min |

Memory: with `max_feature_nodes=None` every active feature is a node, and `attribute()` allocates a
float32 edge matrix of (90,241 + 1) × (90,240 + 37 × 83 + 1) ≈ **33.7 GB** (computed) on the CPU, next
to 16 GB of float32 model weights. In the `batch_size=128` run the system used 33.5 GB of swap on the
64 GB machine (measured). The peak was not measured cleanly, because the runs were stopped first.

## Dtype check (float32 vs bfloat16; the 0.05 limit applies within one dtype)

Max |Δ log-prob| at the final position over the full vocabulary (`circuit/task0_dtype.py`):

| Comparison | Dev (Qwen3-0.6B, 20 prompts) | Main (Qwen3-4B, 5 prompts) |
| --- | --- | --- |
| transformers fp32 vs circuit-tracer fp32 | 6.9e-5 | 1.8e-4 |
| transformers bf16 vs circuit-tracer bf16 | 0.54 | 1.41 |
| transformers fp32 vs transformers bf16 | 0.76 | 0.47 |

## Development model (Qwen3-0.6B, 20 prompts, MPS, float32, defaults, encoders in memory)

Logits 6.9e-5 (pass). About 20,000 active features per prompt. Graph time: mean 212 s, max 295 s.
Replacement score median 0.36, completeness median 0.82. Peak memory: process RSS 21.6 GB, MPS driver
76.9 GB (allocator cache included; the system swapped). One graph with `batch_size=128` (diagnostic):
peak MPS allocated 25.4 GB, driver 33.3 GB, 75 s.

## For the reviewer

The plan allows only offload or lazy loading for G0. Offload (`cpu`) does not save memory on unified
memory, and lazy loading is already on. The two settings that act on the measured problems are
`attribute()`'s `batch_size` (needed on MPS) and `max_feature_nodes` (sets the edge-matrix size and
the time of the feature phase). The library leaves `max_feature_nodes` unlimited by default; using it
would change which features the graphs contain. Any change is a change to `PLAN.md` and is not made here.
