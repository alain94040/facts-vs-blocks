# Experiment D report: do the middle layers share one set of directions?

Models: Qwen2.5-7B-Instruct (main), Qwen2.5-14B-Instruct (confirm). Seed 0 everywhere. Plan:
`basis/PLAN.md`. Numbers are as measured; the only interpretation is the decision table at the end.

Zones (`results/basis/zones.json`):

| Model | ENTRY | FAMILY | OUTPUT |
| --- | --- | --- | --- |
| 7B (28 layers, d = 3,584) | 0–5 | 6–17 | 18–27 |
| 14B (48 layers, d = 5,120) | 0–9 | 10–30 | 31–47 |

The outlier coordinates were taken from Repeat task 4b on 7B, and found the same way in
`results/basis/task3/outliers.json` on 14B.

## Task 0: exactness checks (passed)

`results/basis/task0/checks.json`. The development model (0.5B) was checked on its whole family,
and 7B on layer 10 with a two-layer zone {10, 11}.

| Check (tolerance 1e-9) | 0.5B | 7B |
| --- | --- | --- |
| max \|kept energy at k = d − 1\| | 5.6e-15 | 4.0e-15 |
| min (own − shared), all layers and k | −6.7e-15 | −5.1e-15 |
| max change of own energy under rotation | 1.0e-15 | 1.0e-15 |

## Task 1 (7B) and task 3a (14B): kept energy and sharing efficiency

Full tables: `task1/energy.csv` (7B) and `task3/energy.csv` (14B). They cover every zone, type,
setting, k ∈ {d/16, d/8, d/4, d/2}, and for the family also the other zones' bases. Plots:
`task1/efficiency_<type>.png` and `task3/efficiency_<type>.png`, one line per zone, real solid
and null dashed, with all coordinates (left) and without outliers (right).

### Sharing efficiency at k = d/4, real / null, all coordinates

| Type | 7B ENTRY | 7B FAMILY | 7B OUTPUT | 14B ENTRY | 14B FAMILY | 14B OUTPUT |
| --- | --- | --- | --- | --- | --- | --- |
| q | 0.818 / 0.724 | 0.799 / 0.565 | 0.764 / 0.631 | 0.825 / 0.684 | 0.723 / 0.505 | 0.667 / 0.547 |
| k | 0.853 / 0.794 | 0.794 / 0.634 | 0.808 / 0.706 | 0.809 / 0.694 | 0.672 / 0.499 | 0.671 / 0.543 |
| v | 0.702 / 0.615 | 0.751 / 0.543 | 0.723 / 0.612 | 0.647 / 0.483 | 0.619 / 0.440 | 0.590 / 0.478 |
| o | 0.683 / 0.629 | 0.633 / 0.535 | 0.604 / 0.558 | 0.654 / 0.570 | 0.608 / 0.490 | 0.568 / 0.520 |
| gate | 0.877 / 0.787 | 0.835 / 0.625 | 0.778 / 0.664 | 0.802 / 0.704 | 0.763 / 0.565 | 0.707 / 0.597 |
| up | 0.766 / 0.657 | 0.865 / 0.618 | 0.784 / 0.662 | 0.720 / 0.624 | 0.772 / 0.565 | 0.686 / 0.599 |
| down | 0.785 / 0.738 | 0.764 / 0.675 | 0.749 / 0.706 | 0.662 / 0.620 | 0.697 / 0.599 | 0.672 / 0.620 |
| **attn mean** | 0.764 / 0.691 | **0.745 / 0.569** | 0.725 / 0.627 | 0.734 / 0.608 | **0.656 / 0.483** | 0.624 / 0.522 |
| **mlp mean** | 0.809 / 0.727 | **0.821 / 0.639** | 0.770 / 0.677 | 0.728 / 0.649 | **0.744 / 0.576** | 0.688 / 0.605 |

The group means are unweighted averages over the group's types. Without the outlier coordinates,
every number in this table changes by at most 0.017 (ENTRY gate null, both models). Across all
k the largest change is 0.034 (7B ENTRY gate null at d/16). See the CSVs.

The own-basis energy of `k` and `v` is 1.000 at d/4 on both models. Their other side is 512
(7B) or 1,024 (14B), smaller than d/4, so for these two types efficiency equals shared energy.

### Family layers, 7B, all coordinates: kept energy across k

| Type | k/d | own | shared | null | eff real | eff null |
| --- | --- | --- | --- | --- | --- | --- |
| gate | 1/16 · 1/8 · 1/4 · 1/2 | 0.251 · 0.382 · 0.576 · 0.823 | 0.181 · 0.298 · 0.482 · 0.755 | 0.119 · 0.206 · 0.360 · 0.618 | 0.719 · 0.776 · 0.835 · 0.915 | 0.465 · 0.536 · 0.625 · 0.751 |
| up | | 0.187 · 0.323 · 0.533 · 0.808 | 0.150 · 0.270 · 0.464 · 0.754 | 0.093 · 0.177 · 0.330 · 0.597 | 0.790 · 0.825 · 0.865 · 0.931 | 0.492 · 0.545 · 0.618 · 0.738 |
| down | | 0.170 · 0.285 · 0.468 · 0.732 | 0.115 · 0.203 · 0.358 · 0.620 | 0.090 · 0.170 · 0.316 · 0.577 | 0.677 · 0.714 · 0.764 · 0.847 | 0.532 · 0.596 · 0.675 · 0.788 |
| q | | 0.529 · 0.704 · 0.867 · 0.974 | 0.346 · 0.509 · 0.694 · 0.870 | 0.176 · 0.302 · 0.491 · 0.742 | 0.653 · 0.720 · 0.799 · 0.893 | 0.329 · 0.426 · 0.565 · 0.761 |
| k | | 0.840 · 0.993 · 1.000 · 1.000 | 0.415 · 0.602 · 0.794 · 0.944 | 0.236 · 0.401 · 0.634 · 0.880 | 0.492 · 0.606 · 0.794 · 0.944 | 0.277 · 0.403 · 0.634 · 0.880 |
| v | | 0.691 · 0.968 · 1.000 · 1.000 | 0.345 · 0.540 · 0.751 · 0.926 | 0.179 · 0.322 · 0.543 · 0.819 | 0.495 · 0.557 · 0.751 · 0.926 | 0.256 · 0.332 · 0.543 · 0.819 |
| o | | 0.341 · 0.530 · 0.757 · 0.946 | 0.178 · 0.297 · 0.480 · 0.735 | 0.127 · 0.230 · 0.405 · 0.674 | 0.523 · 0.561 · 0.633 · 0.778 | 0.371 · 0.435 · 0.535 · 0.712 |

### Family layers with other zones' bases, k = d/4, all coordinates (sharing efficiency)

| Type | 7B own zone | 7B ENTRY basis | 7B OUTPUT basis | 14B own zone | 14B ENTRY basis | 14B OUTPUT basis |
| --- | --- | --- | --- | --- | --- | --- |
| q | 0.799 | 0.496 | 0.569 | 0.723 | 0.409 | 0.455 |
| k | 0.794 | 0.391 | 0.450 | 0.672 | 0.329 | 0.374 |
| v | 0.751 | 0.373 | 0.371 | 0.619 | 0.307 | 0.261 |
| o | 0.633 | 0.398 | 0.392 | 0.608 | 0.389 | 0.379 |
| gate | 0.835 | 0.548 | 0.615 | 0.763 | 0.506 | 0.514 |
| up | 0.865 | 0.591 | 0.697 | 0.772 | 0.546 | 0.559 |
| down | 0.764 | 0.568 | 0.581 | 0.697 | 0.493 | 0.516 |

## Task 2 (7B) and task 3b (14B): editing the family

Every family layer's matrices of one group are projected to rank k on the residual side.
Evaluation uses Experiment C's SEARCH halves (TEXT 100 chunks, REASON, RECALL-HARD), with paired
bootstrap 95% intervals against base. Base on 7B: REASON accuracy 0.380, RECALL-HARD accuracy
0.939. Base on 14B: 0.550 and 0.974. Compression is the family's original parameter count for
the group divided by the scheme's. Full tables: `task2/edits.csv` and `task3/edits.csv`. Plots:
`task2/text_vs_compression.png` and `task3/text_vs_compression.png`.

### 7B

| Config | Group | k | Compression | ΔTEXT (nats) [95% CI] | REASON acc | RECALL-HARD acc |
| --- | --- | --- | --- | --- | --- | --- |
| own | mlp | d/8 | 6.73 | +2.412 [+2.329, +2.490] | 0.010 | 0.017 |
| own | mlp | d/4 | 3.36 | +2.002 [+1.943, +2.066] | 0.020 | 0.000 |
| own | mlp | d/2 | 1.68 | +1.266 [+1.223, +1.307] | 0.030 | 0.183 |
| shared | mlp | d/8 | 7.88 | +2.573 [+2.489, +2.667] | 0.020 | 0.009 |
| shared | mlp | d/4 | 3.94 | +1.995 [+1.944, +2.056] | 0.005 | 0.009 |
| shared | mlp | d/2 | 1.97 | +1.698 [+1.648, +1.755] | 0.015 | 0.087 |
| null | mlp | d/8 | 7.88 | +2.769 [+2.690, +2.855] | 0.020 | 0.009 |
| null | mlp | d/4 | 3.94 | +2.317 [+2.245, +2.393] | 0.005 | 0.000 |
| null | mlp | d/2 | 1.97 | +1.974 [+1.918, +2.031] | 0.020 | 0.000 |
| own | attn | d/8 | 2.91 | +0.689 [+0.652, +0.728] | 0.015 | 0.643 |
| own | attn | d/4 | 1.45 | +0.137 [+0.125, +0.149] | 0.250 | 0.870 |
| own | attn | d/2 | 0.73 | +0.009 [+0.004, +0.014] | 0.330 | 0.939 |
| shared | attn | d/8 | 6.98 | +1.303 [+1.262, +1.345] | 0.020 | 0.052 |
| shared | attn | d/4 | 3.49 | +0.751 [+0.718, +0.786] | 0.010 | 0.209 |
| shared | attn | d/2 | 1.75 | +0.127 [+0.112, +0.141] | 0.100 | 0.843 |
| null | attn | d/8 | 6.98 | +1.773 [+1.729, +1.819] | 0.025 | 0.009 |
| null | attn | d/4 | 3.49 | +1.513 [+1.470, +1.563] | 0.030 | 0.017 |
| null | attn | d/2 | 1.75 | +0.616 [+0.584, +0.647] | 0.010 | 0.748 |

`own` attention at d/2 has a compression ratio of 0.73, below 1. Under the plan's count, d × k + k × n
exceeds d × n for `k` and `v`, whose n is 512.

### 14B (base, own(d/4), shared(d/4))

| Config | Group | Compression | ΔTEXT (nats) [95% CI] | REASON acc | RECALL-HARD acc |
| --- | --- | --- | --- | --- | --- |
| own | mlp | 2.92 | +2.133 [+2.050, +2.217] | 0.030 | 0.009 |
| shared | mlp | 3.93 | +3.268 [+3.167, +3.370] | 0.005 | 0.000 |
| own | attn | 1.50 | +0.214 [+0.199, +0.230] | 0.260 | 0.983 |
| shared | attn | 3.71 | +0.996 [+0.959, +1.034] | 0.010 | 0.678 |

### Implementation note: one fix after the first run

The first version of `basis/task2.py` worked in these steps:

1. Fold the RMSNorm gain into the reading matrices.
2. Project the folded matrices.
3. Divide the gain out again before storing in float16.

Some 14B gains are close to zero; the smallest is 2.4e-7 in layer 10. Dividing by them pushed the
stored `k` weights past the float16 maximum, so the 14B `shared(d/4)` attention run gave a NaN.
The edit now stores the folded projection directly and sets that group's norm gain to 1. The
model computes the same function, U Uᵀ W_folded. With U = I, the fixed edit reproduces the base
logits within float16 rounding (max |Δlogit| ≤ 0.09 on logits up to 36, 0.5B and 7B).

Both 7B task 2 and the 14B check were rerun with the fix (commit `6f2b52a`). The 7B rows and the
three finite 14B rows agree with the first run to 3 decimals. The NaN became +0.996. The tables
above come from the rerun.

On the 0.5B development model, `own(d/2)` and `shared(d/2)` of the MLP gave the same mean TEXT
change (+2.2689 vs +2.2689). This was checked and is a coincidence of averaging:

- The edited weights differ: on layer 5 `up`, ‖own − shared‖_F = 24.6 against ‖W‖_F = 52.5, and the
  kept energy is 0.776 vs 0.702.
- The per-chunk TEXT losses differ by 0.22 nats on average (correlation 0.83).
- The REASON log-probability differs (−8.43 vs −5.71).

## Decision table

Thresholds as written in `basis/PLAN.md`; "family efficiency" uses the group mean of the types
at k = d/4 with all coordinates. The no-outlier setting gives the same marks.

| Result | 7B | 14B |
| --- | --- | --- |
| Task 0 checks fail | **Does not apply** (all pass, errors ≤ 7e-15) | — |
| Family efficiency at d/4 ≥ 0.9 and null ≤ 0.7, MLP | **Does not apply**: 0.821 (no single type reaches 0.9: gate 0.835, up 0.865, down 0.764); null 0.639 | **Does not apply**: 0.744; null 0.576 |
| Same for attention | **Does not apply**: 0.745; null 0.569 | **Does not apply**: 0.656; null 0.483 |
| Family efficiency above OUTPUT by ≥ 0.1 at d/4 | **Does not apply**: MLP +0.051, attention +0.020; largest single type up +0.081; k −0.014 | **Does not apply**: MLP +0.056, attention +0.032; largest up +0.086 |
| Other zones' bases keep the family's energy as well as its own | **Does not apply**: 0.37–0.70 with the ENTRY or OUTPUT basis against 0.63–0.87 with the family's own | **Does not apply**: 0.26–0.56 against 0.61–0.77 |
| `shared(k)` costs ≤ 0.02 TEXT nats more than `own(k)` | **Does not apply** in 5 of 6 cases: MLP +0.161 (d/8), +0.432 (d/2); attention +0.614, +0.614, +0.118. Holds in 1 case, MLP d/4 (−0.007), where both are above +1.99 nats | **Does not apply**: MLP +1.135, attention +0.782 |
| Some `shared(k)` keeps TEXT within +0.05 of base at compression ≥ 2 | **Does not apply**: the smallest `shared` change is +0.127 (attention d/2, compression 1.75); at compression ≥ 2 the smallest is +0.751 | **Does not apply**: +0.996 and +3.268 |
| `shared(k)` damage close to `null(k)` | **Does not apply on TEXT**: `shared` is lower than `null` in all 6 pairs, with disjoint 95% CIs (MLP by 0.20–0.32 nats, attention by 0.47–0.76). For MLP, REASON and RECALL-HARD accuracies of both are at or near 0 at every k. | Not measured (`null` is not in the task 3 subset) |
| 14B rows agree with 7B rows | **Applies**: every row measured on both models gets the same mark. The 14B efficiencies are lower than 7B's in every zone and group. | |

## Limits that bear on these numbers

- These are energy measurements on weights. Task 2 shows how they translate to loss: on both
  models, the MLP edits at every k tested cost more than 1.2 nats of TEXT even with each layer's
  own basis.
- The sharing efficiency of `k` and `v` at d/4 and d/2 equals their shared energy, because their
  own energy is already 1 (their other side is smaller than k).
- No training was done (plan section 4).
