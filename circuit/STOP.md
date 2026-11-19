# STOP: gate G0 fails on the main model with the revision 3 graph settings (circuit task 0)

Gate G0 (circuit/PLAN.md revision 3), main model `Qwen/Qwen3-4B` with `mwhanna/qwen3-4b-transcoders`,
5 PROG P1 prompts (seeds 9001–9005). The graph settings are float32, lazy encoder, safetensors 0.7.0
with `circuit/ct_compat.py`, and `batch_size = 64`. The gate requires: logits match within 0.05 in one
dtype, one graph ≤ 20 minutes, peak memory ≤ 48 GB. At `max_feature_nodes = 8192` and again at 4096,
logits and graph time pass, and **memory fails**. Per the plan no other setting was tried, and Task 1
was not started. All numbers are in `results/circuit/task0/metrics.json` (key `revision3`).

Peak memory is the peak of process RSS + MPS driver-allocated memory, sampled every 0.5 s (the
measure fixed in `circuit/task0.py` before these runs). The MPS driver figure includes the PyTorch
MPS allocator's cache.

## Main model

| `max_feature_nodes` | Graphs | Logits (max abs diff) | Seconds per graph | Peak memory (GB): RSS + MPS driver = sum | MPS memory held by tensors (GB) | System swap (GB) | Completeness (median) | Replacement (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 1 of 5 (stopped: peak already over the limit) | 1.8e-4 | 756 | 36.6 + 75.5 = **90.9** | 29.4 | 6.4 → 29.4 | 0.833 | 0.333 |
| 4096 | 5 of 5 | 1.8e-4 | mean 407, max 518 | 37.1 + 74.7 = **85.0** | 30.1 | 9.5 → 30.0 | 0.806 | 0.261 |

At 4096, per prompt (seed: seconds, completeness, active features, pruned nodes): 9001: 518, 0.806,
90,240, 2,612; 9002: 328, 0.803, 93,496, 2,538; 9003: 396, 0.810, 93,856, 2,475; 9004: 409, 0.807,
94,388, 2,598; 9005: 384, 0.800, 92,681, 2,634.

Where the peak comes from (4096 run, measured):
- right after loading, before any graph: RSS 37.1 GB, MPS driver 15.6 GB, MPS tensors 15.1 GB;
- during graphs: MPS tensors peak at 30.1 GB, while the MPS driver holds 74.7 GB.

RSS never rose above its post-load value. The MPS driver figure is about 45 GB above what tensors
hold. System swap grew by about 20 GB during the graphs, so the machine was under real memory
pressure.

## Development model (Qwen3-0.6B, 20 prompts, MPS, revision 3 settings, encoders in memory)

| Run | Logits | Seconds per graph | Peak (GB): RSS + driver = sum | Completeness (median) | Replacement (median) |
| --- | --- | --- | --- | --- | --- |
| `max_feature_nodes = 8192` | 6.9e-5 | mean 63, max 65 | 13.0 + 28.2 = 40.8 | 0.816 | 0.357 |
| no cap (for step 5) | 6.9e-5 | mean 78, max 81 | 23.0 + 28.2 = 50.7 | 0.819 | 0.362 |

**Step 5, effect of the cap** (reported, not gated; `results/circuit/task0/cap_effect/`). Median
Jaccard overlap of the pruned graphs' feature sets: 0.882 by node (layer, position, feature), 0.908
by (layer, feature). Median pruned feature nodes: 1,382 capped, 1,572 uncapped. The first uncapped
run saved no feature sets, so the uncapped graphs were rebuilt with `batch_size = 64`. Compared with
the first run (batch 512), completeness differs by at most 1.2e-7 and replacement by at most 6e-8,
and the pruned node counts are identical on all 20 prompts.

## For the reviewer

Measured, not acted on: the main-model peak is set by (a) 37 GB of process RSS left after loading the
model through TransformerLens, which converts the Hugging Face weights in CPU memory, and (b) the MPS
allocator's cache, about 45 GB above the memory tensors hold. The cap changes neither much: 8192 and
4096 give 90.9 and 85.0 GB. Graph time is within the limit at both caps. The budget rule (more than
5 minutes per graph → 20 items instead of 40) would apply at 4096 (mean 6.8 min).
