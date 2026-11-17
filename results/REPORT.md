# Facts vs building blocks: Phase 2 report

Numbers as measured. No interpretation beyond marking which rows of the `PLAN.md` decision table
apply. All analysis scripts were developed on the pilot `P2-main-2k-s0` (and `P1-main-2k-s0` where
two models were needed) and then run once on the Task 2 models. Pilot outputs are in
`results/task3_pilot`, `task4_pilot` and `task5_pilot`. Method choices not fixed by `PLAN.md` were
asked and answered before the real runs (`QUESTIONS.md`, "Task 5" section). One concern was raised
after the real run, and the method was not changed (`QUESTIONS.md`, last section).

Commits: Task 3 `e57c2e6`, Task 4 `a503527`, Task 5 `4389aad`.

## Task 2 (Phase 1): models

Gate 2 passed for all 11 runs (`results/task2/summary.csv`).

| Run | Story loss | Recall, training templates | Template 5, held out | Name-copy |
| --- | --- | --- | --- | --- |
| main-0 s0 / s1 | 1.692 / 1.699 | - | - | 0.000 (not gated) |
| main-2k s0 / s1 | 1.692 / 1.693 | 0.960 / 0.951 | 0.940 / 0.927 | 0.897 / 0.853 |
| main-8k s0 / s1 | 1.696 / 1.696 | 0.973 / 0.970 | 0.961 / 0.959 | 1.000 / 1.000 |
| main-32k s0 / s1 | 1.747 / 1.747 | 0.688 / 0.773 | 0.685 / 0.763 | 0.999 / 1.000 |
| ctrl-8k-x4 s0 | 1.693 | 0.997 | 0.994 | 1.000 |
| fork-A s0 | 1.701 | 0.987 | 0.982 | 1.000 |
| fork-B s0 | 1.700 | 0.983 | 0.979 | 1.000 |

## Task 3: which parts hold the facts

Script: `analysis/task3.py`. Per-run results are in `results/task3/<run>/metrics.json`, and the
summary is in `results/task3/summary.csv`.

Setup:

- **FACT:** templates 1 to 4 for the first 1,000 people of the run's set (24,000 prompts).
- **FORMAT:** 1,000 unseen set-B people, 52,862 scored tokens.
- **STORY:** 2,000 validation rows.
- **Mean reference:** the first 1,000 STORY rows plus the FORMAT biographies (334,047 positions).
- **Random baseline:** 3 draws (seed 0).

A gradient check on the pilot matched finite differences to 4 significant digits.

**Knowledge set.** K is the smallest k on the grid 8, 16, ..., 4096 at which FACT recall falls to
half its unablated value or below. Loss changes are in nats.

| Run | People | K | Recall: base → at K | Recall, K random | STORY Δ at K | STORY Δ, K random | FORMAT Δ at K | FORMAT Δ, K random | Name-copy: base → at K |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| main-2k s0 | 2,000 | 256 | 0.960 → 0.245 | 0.85–0.89 | +0.168 | +0.043 | +0.514 | +0.015 | 0.897 → 0.271 |
| main-2k s1 | 2,000 | 256 | 0.951 → 0.284 | 0.84–0.87 | +0.159 | +0.044 | +0.617 | +0.016 | 0.853 → 0.207 |
| main-8k s0 | 8,000 | 128 | 0.972 → 0.443 | 0.86–0.92 | +0.139 | +0.018 | +0.188 | +0.001 | 1.000 → 0.995 |
| main-8k s1 | 8,000 | 256 | 0.970 → 0.245 | 0.78–0.82 | +0.171 | +0.047 | +0.178 | +0.013 | 1.000 → 0.943 |
| main-32k s0 | 32,000 | 64 | 0.685 → 0.282 | 0.65–0.67 | +0.251 | +0.010 | +0.566 | +0.002 | 0.999 → 0.782 |
| main-32k s1 | 32,000 | 128 | 0.774 → 0.324 | 0.65–0.70 | +0.173 | +0.022 | +0.815 | +0.002 | 1.000 → 0.333 |
| ctrl-8k-x4 s0 | 8,000 | 256 | 0.998 → 0.297 | 0.89–0.92 | +0.340 | +0.045 | +0.591 | +0.014 | 1.000 → 0.341 |
| fork-A s0 | 8,000 | 8 | 0.987 → 0.481 | 0.985–0.986 | +0.069 | +0.001 | +0.168 | 0.000 | 1.000 → 0.995 |
| fork-B s0 | 8,000 | 8 | 0.984 → 0.445 | 0.979–0.982 | +0.064 | +0.001 | +0.210 | 0.000 | 1.000 → 0.968 |

**Knowledge set by layer, and overlap.** Jaccard overlap is between the knowledge set and the top
K neurons by I_STORY, and by I_FORMAT.

| Run | Neurons per layer (L0 … L5) | Jaccard with top-K I_STORY | Jaccard with top-K I_FORMAT | Jaccard, two random sets |
| --- | --- | --- | --- | --- |
| main-2k s0 | 173, 19, 23, 12, 18, 11 | 0.089 | 0.156 | 0.021 |
| main-2k s1 | 173, 29, 14, 11, 17, 12 | 0.062 | 0.148 | 0.021 |
| main-8k s0 | 111, 3, 6, 3, 1, 4 | 0.041 | 0.118 | 0.011 |
| main-8k s1 | 200, 18, 7, 13, 12, 6 | 0.092 | 0.145 | 0.021 |
| main-32k s0 | 44, 6, 3, 4, 2, 5 | 0.103 | 0.058 | 0.005 |
| main-32k s1 | 96, 6, 5, 9, 7, 5 | 0.103 | 0.076 | 0.011 |
| ctrl-8k-x4 s0 | 162, 15, 17, 18, 12, 32 | 0.082 | 0.118 | 0.021 |
| fork-A s0 | 8, 0, 0, 0, 0, 0 | 0.143 | 0.000 | 0.001 |
| fork-B s0 | 8, 0, 0, 0, 0, 0 | 0.067 | 0.067 | 0.001 |

**Single-head ablations.** The two largest recall drops per run (change in recall; change in
STORY loss). The last column counts heads whose ablation alone removes more than half of base
recall.

| Run | Largest drop | Second | Heads removing > half of recall |
| --- | --- | --- | --- |
| main-2k s0 | L0H0 −0.428; +0.017 | L0H5 −0.335; +0.025 | 0 |
| main-2k s1 | L0H6 −0.434; +0.024 | L0H4 −0.378; +0.023 | 0 |
| main-8k s0 | L0H4 −0.918; +0.118 | L0H1 −0.554; +0.019 | 2 |
| main-8k s1 | L0H2 −0.720; +0.101 | L0H4 −0.355; +0.025 | 1 |
| main-32k s0 | L0H4 −0.637; +0.071 | L0H5 −0.556; +0.069 | 4 |
| main-32k s1 | L0H3 −0.657; +0.007 | L0H7 −0.657; +0.053 | 3 |
| ctrl-8k-x4 s0 | L0H4 −0.932; +0.041 | L0H1 −0.777; +0.018 | 2 |
| fork-A s0 | L3H7 −0.663; +0.042 | L1H0 −0.546; +0.107 | 2 |
| fork-B s0 | L1H0 −0.624; +0.108 | L3H7 −0.620; +0.042 | 2 |

**Effective rank** of `mlp.up` (L0 … L5) ranges from 216 to 239 out of 256 in every run, and of
`mlp.down` from 236 to 244. Per-matrix values are in each run's `metrics.json`.

**Plots:**

- Ablation curves: `results/task3/<run>/ablation_curves.png`
- K against number of people: `results/task3/K_vs_people.png`
- Knowledge set by layer: `results/task3/knowledge_by_layer.png`
- Effective rank: `results/task3/effective_rank_up.png` and `results/task3/<run>/effective_rank.png`

**Predictions in `PLAN.md` against measurements:**

| Prediction if the split is real | Measured |
| --- | --- |
| K grows roughly in proportion to people, 2k → 32k | K = 256 / 256 (2k), 128 / 256 (8k), 64 / 128 (32k) |
| ctrl-8k-x4 has K near main-8k, not main-32k | ctrl-8k-x4 K = 256; main-8k 128 / 256; main-32k 64 / 128 |
| Knowledge-set ablation raises STORY and FORMAT loss no more than K random neurons | STORY +0.06 to +0.34 against +0.00 to +0.05 random; FORMAT +0.17 to +0.82 against ≤ +0.02 random |
| Overlaps near the random expectation | 0.04 to 0.16 against 0.001 to 0.021 random (fork-A FORMAT: 0.000) |
| No single head ablation removes most of the recall | 0 heads in main-2k; 1 to 4 heads in every other run |

## Task 4: swapping parts between models

Script: `analysis/task4.py`. Results are in `results/task4/interpolation.json` and
`results/task4/swaps.json`, with the plot in `results/task4/swaps.png`. FACT recall uses the first
1,000 people of each set.

**Interpolation (STORY loss at α = 0, .25, .5, .75, 1):**

| Path | Losses | Barrier |
| --- | --- | --- |
| fork-A → fork-B | 1.703, 1.705, 1.708, 1.704, 1.702 | 0.006 |
| fork-A → main-8k s1 | 1.703, 4.076, 7.078, 3.985, 1.699 | 5.377 |

The fork-A to fork-B barrier is below 0.1 nats, so no hybrid row is flagged.

**Hybrids.** Columns give base fork-A / base fork-B. The originals are:

| Model | Recall, own people | Recall, other set | STORY loss | FORMAT loss |
| --- | --- | --- | --- | --- |
| fork-A | 0.987 | 0.005 | 1.703 | 0.300 |
| fork-B | 0.984 | 0.005 | 1.702 | 0.299 |

| Hybrid | Parts from donor | Recall, base people | Recall, donor people | STORY loss | FORMAT loss |
| --- | --- | --- | --- | --- | --- |
| H1 | all attention | 0.121 / 0.115 | 0.005 / 0.004 | 1.749 / 1.750 | 0.350 / 0.380 |
| H2 | all MLP | 0.029 / 0.035 | 0.004 / 0.005 | 1.766 / 1.766 | 0.395 / 0.363 |
| H3 | MLP layer 0 (holds the whole knowledge set, K = 8, in both forks) | 0.762 / 0.762 | 0.004 / 0.005 | 1.728 / 1.728 | 0.314 / 0.304 |
| H4 | MLP layer 0 | 0.762 / 0.762 | 0.004 / 0.005 | 1.728 / 1.728 | 0.314 / 0.304 |
| H5 | MLP layer 1 | 0.812 / 0.825 | 0.005 / 0.005 | 1.729 / 1.727 | 0.302 / 0.306 |
| H6 | MLP layer 2 | 0.721 / 0.706 | 0.004 / 0.005 | 1.732 / 1.731 | 0.312 / 0.308 |
| H7 | MLP layer 3 | 0.269 / 0.291 | 0.004 / 0.004 | 1.739 / 1.738 | 0.313 / 0.310 |
| H8 | MLP layer 4 | 0.248 / 0.246 | 0.003 / 0.004 | 1.742 / 1.740 | 0.302 / 0.311 |
| H9 | MLP layer 5 | 0.480 / 0.461 | 0.004 / 0.005 | 1.736 / 1.735 | 0.307 / 0.307 |
| H10 | token embeddings (tied with output) | 0.006 / 0.005 | 0.017 / 0.017 | 1.730 / 1.728 | 0.323 / 0.307 |
| H1 control | all attention from main-8k s1 | 0.000 / 0.000 | 0.000 / 0.000 | 6.870 / 6.741 | 10.000 / 10.375 |
| H2 control | all MLP from main-8k s1 | 0.000 / 0.000 | 0.000 / 0.000 | 10.679 / 10.308 | 10.835 / 10.938 |

**Expectations in `PLAN.md` against measurements:**

- **H1** (base facts kept above 80% of original, STORY up less than 0.1 nats): base recall is 12%
  of original; STORY is up 0.046 / 0.048.
- **H2** (base facts lost, donor facts appear): base recall is 0.03; donor recall is 0.004 / 0.005.
- **H3** (facts swap, little story damage): base recall is 0.762; donor recall is 0.004 / 0.005;
  STORY is up 0.025 / 0.026.
- **Control** (damage far more): STORY loss is 6.7 to 10.7 against 1.75 to 1.77 for H1 and H2.

## Task 5: do the same components recur

Script: `analysis/task5.py`. Results are in `results/task5/heads.csv` (every head of the 11 trained
models plus the untrained weights), `head_matching.json` and `neuron_matching.json`. Plots are
`head_recurrence.png` and `neuron_recurrence.png`.

Feature sets are as agreed in `QUESTIONS.md`:

- **FULL:** 41 standardised features after dropping the constant first singular values.
- **STORY:** 36 features.

**Head matching.** Recurrence uses a per-head null from 100 column shuffles, at the 5th percentile.

| Pair | Features | Fraction of heads that recur | Median nearest distance | Median null 5th percentile | Nearest head in the same layer |
| --- | --- | --- | --- | --- | --- |
| main-0 s0 vs s1 | STORY | 1.000 | 2.06 | 5.14 | 0.458 |
| main-2k s0 vs s1 | FULL | 1.000 | 2.32 | 4.91 | 0.521 |
| main-8k s0 vs s1 | FULL | 0.979 | 2.45 | 4.79 | 0.354 |
| main-32k s0 vs s1 | FULL | 1.000 | 2.63 | 5.15 | 0.542 |
| main-2k s0 vs main-32k s0 | FULL | 1.000 | 3.52 | 6.05 | 0.417 |

The pilot pair (P2 vs P1, two recipes with the same initialisation seed) also gave 1.000.
`QUESTIONS.md` records the concern that the column-shuffle null may be weak.

**Pooled heads, k-means (8 clusters):**

- **FULL features, 9 biography models (432 heads):** every one of the 8 clusters contains heads
  from all 9 models. Cluster sizes are 23 to 100. The 23-head cluster has a mean FORMAT induction
  score of 0.76; the others have 0.08 to 0.32.
- **STORY features, 11 trained models plus the untrained model (576 heads):** 7 clusters each
  contain heads from all 11 trained models and none from the untrained model. The eighth cluster
  holds exactly the 48 heads of the untrained model.

Per-cluster mean features are in `head_matching.json`.

**Neuron matching** compares seed 0 with seed 1 on fixed positions: 200,000 STORY positions for
main-0, and 78,047 FORMAT plus 121,953 STORY positions for the others. The null is seed 0 against
the untrained model on the same positions.

| Run | Fraction > 0.5, seed 1 | Fraction > 0.8, seed 1 | Fraction > 0.5, null | Fraction > 0.8, null |
| --- | --- | --- | --- | --- |
| main-0 | 0.025 | 0.003 | 0.000 | 0.000 |
| main-2k | 0.054 | 0.003 | 0.000 | 0.000 |
| main-8k | 0.077 | 0.008 | 0.000 | 0.000 |
| main-32k | 0.071 | 0.010 | 0.000 | 0.000 |

| Run | Inside knowledge set: n, > 0.5, > 0.8 | Outside: > 0.5, > 0.8 |
| --- | --- | --- |
| main-2k s0 | 256, 0.246, 0.027 | 0.046, 0.002 |
| main-8k s0 | 128, 0.305, 0.047 | 0.072, 0.007 |
| main-32k s0 | 64, 0.625, 0.203 | 0.065, 0.008 |

By layer (fraction > 0.5, seed 1, L0 … L5):

- main-0: 0.037, 0.019, 0.025, 0.006, 0.024, 0.038
- main-2k: 0.078, 0.021, 0.022, 0.026, 0.073, 0.104
- main-8k: 0.056, 0.027, 0.043, 0.096, 0.090, 0.150
- main-32k: 0.080, 0.036, 0.031, 0.069, 0.080, 0.128

The null is 0.002 to 0.003 in layer 0 and 0.000 elsewhere.

**Predictions in `PLAN.md` against measurements:**

| Prediction | Measured |
| --- | --- |
| Most heads recur across seeds and fact loads | 0.98 to 1.00 by the planned criterion (see the null concern) |
| Knowledge-set neurons recur less than the others | Above 0.5: 0.25 / 0.30 / 0.63 inside against 0.05 / 0.07 / 0.07 outside |
| The null gives low recurrence | Neurons: ≤ 0.003. Heads: the null is the shuffle and gives 5% by construction |

## Decision table

| Result | Applies? | Basis |
| --- | --- | --- |
| K grows with people, ctrl-8k-x4 near main-8k, knowledge-set ablation spares stories and format, H1 keeps facts, H3 moves facts → split is real | **No** | K does not grow; H1 keeps 12% of base facts; H3 moves no donor facts (0.004) |
| Knowledge-set ablation damages stories or format as much as it damages facts, or H1 and H3 damage everything → facts and procedures share weights | **Not marked: needs reviewer** | H1 and H3 do not damage everything (STORY +0.03 to +0.05). For the first clause, `PLAN.md` gives no common unit for "as much as" between a recall halving and a loss change. Measured: at K, STORY +0.06 to +0.34 and FORMAT +0.17 to +0.82, against ≤ +0.05 for random sets |
| K does not grow with people, or ctrl-8k-x4 looks like main-32k → the measure tracks text, not facts | **Yes** | K = 256 / 256, 128 / 256, 64 / 128 for 2k, 8k and 32k |
| Heads recur across seeds well above the null → reusable attention blocks exist | **Yes, by the planned criterion** | 0.98 to 1.00 against 0.05. The null's strength is questioned in `QUESTIONS.md` |
| Recurrence no better than the null → no shared blocks | No | |
| Fork barrier above 0.1 nats → swaps not interpretable | No | Barrier 0.006 |
| Gate 2 failed | No | All runs passed |

**End of Phase 2. Stopped for human review.**
