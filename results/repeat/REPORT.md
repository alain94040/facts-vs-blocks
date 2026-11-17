# Experiment C: repeating layers of a real model, report

Numbers as measured. No interpretation beyond marking which rows of the `repeat/PLAN.md` table
apply.

**Setup:**

- **Models:** main `Qwen/Qwen2.5-7B-Instruct` (28 layers), float16 on MPS. The confirmation model
  is `Qwen/Qwen2.5-14B-Instruct` (48 layers), used in task 5b only. Every script was first run on
  `Qwen/Qwen2.5-0.5B-Instruct`; those outputs are in `dev/` folders and are not results.
- **Selection:** every choice was made on the SEARCH halves. REPORT was first used in task 5.
- **Statistics:** paired 95% bootstrap intervals over items (1,000 resamples, seed 0). "*" marks
  an interval that excludes zero.
- **Commits:** task 0 `92d8f19` through task 5b `c6123f7`.

**Departures from the plan, all reported earlier:**

- **Libraries:** `transformers` and `safetensors` are used to load Qwen (agreed; pinned in
  `repeat/requirements.txt`).
- **Float32 combinations:** `gain` and `partial` combine their two terms in float32, which makes
  `gain(1)`, `partial(0)` and `partial(1)` exact in float16.
- **RECALL size:** 261 items (161 capitals and 100 element symbols), not about 300. Disputed,
  recently moved and multiply spelled capitals were removed.
- **RECALL-HARD size:** 230 items, not 236. The three example elements are excluded from the
  items, as the plan says.
- **Mid-depth reference:** the "selection-free reference" layer at 50% of depth (layer 14 on 7B,
  layer 24 on 14B) is the same layer that task 3 selected (or its mapping). Its rows therefore
  repeat the selected-layer rows.

## Task 0 and 0b: exactness, speed, difficulty, baselines (SEARCH)

| | 0.5B (development) | 7B (main) | 14B (task 5b) |
| --- | --- | --- | --- |
| `base` vs library logits | 0.0 (float32 and float16) | 0.0 | 0.0 |
| `gain(1)`, `partial(0)`, `partial(1)` vs `repeat`, `swap(i,i)`, block `gain(1)` | float32 ≤ 7e-5; float16 0.0 | 0.0 | 0.0 |
| Tokens per second | 30,340 | 3,669 (13.8 s per configuration) | 1,850 |

**Baselines on SEARCH (7B):**

- **REASON:** k = 3 gives 0.38 accuracy, inside the required 30–70% band, so k = 3 is frozen;
  log-probability −2.51.
- **TEXT:** loss 2.604.
- **RECALL:** accuracy 1.00, log-probability −0.026.
- **RECALL-HARD:** accuracy 0.939 (name → number 1.00, number → name 0.88), log-probability −0.25.
  This is above 0.9; as the plan says, it is reported and the set was not changed.

## Task 1: single layers (SEARCH)

Results are in `task1/layers.csv`, with plots `task1/change_<set>.png` (axes clipped; off-scale
points marked). The table lists the layers where the change counts:

| Intervention | REASON log-prob up* | TEXT loss down* | RECALL-HARD log-prob up* |
| --- | --- | --- | --- |
| `repeat` | 0, 1, 2, 6, 7, 9, 10, 14, 25 | none | none |
| `gain(2)` | 0, 1, 2, 5, 7, 9 | none | none |
| `gain(0)` | 4 | none | none |
| `attn2` | 0, 1, 2, 4, 9, 10, 15, 23, 25, 27 | 7, 10, 11 | none |
| `mlp2` | 2, 6, 7, 8, 9 | none | none |

- **Largest REASON gain under `repeat`:** layer 14 (+0.40).
- **REASON accuracy rises significantly in four cases:** `repeat` at layers 0 and 9 (+0.055,
  +0.075), `gain(2)` at layer 0 (+0.075) and `attn2` at layer 4 (+0.045).
- **Late layers hurt everything:** from layer 16 on, `repeat` and `gain(2)` lower REASON, and
  layers 22 to 27 hurt every set.

## Task 2: blocks (SEARCH)

Results are in `task2/blocks.csv`, with heat maps `task2/heat_repeat.png` and `heat_gain2.png`.
Change in REASON log-probability under `repeat`:

| Start → | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Length 2 | +0.32* | +0.43* | +0.23 | −0.09 | +0.33* | −0.26 | −1.45* | −0.56* | −0.41* |
| Length 4 | +0.43* | +0.49* | +0.23 | +0.22 | +0.31* | −0.38* | −0.36* | −0.29 | +0.06 |
| Length 8 | +0.32* | +0.29 | +0.29* | −0.13 | −0.72* | −0.38* | +0.16 | −0.74* | – |

- **Largest gain:** block 8–11, +0.49. Its other changes are TEXT −0.004, RECALL-HARD −0.011 and
  REASON accuracy +0.005, none of them significant.
- **`gain(2)` raises REASON on only four blocks:** 6–7 (+0.61), 6–9 (+0.34), 22–25 (+0.18) and
  18–25 (+0.16).

## Task 3: dose (SEARCH)

Results are in `task3/dose.csv`, with the plot `task3/dose.png`.

| Target | Configuration | REASON log-prob | REASON accuracy | TEXT loss | RECALL-HARD log-prob |
| --- | --- | --- | --- | --- | --- |
| Layer 14 | `partial(0.25 / 0.5 / 0.75 / 1)` | +0.18* / +0.34* / +0.43* / +0.40* | −0.005 / −0.010 / −0.025 / −0.035 | 0.000 / +0.003 / +0.010* / +0.021* | −0.005 / −0.015 / −0.030* / −0.047* |
| Layer 14 | `gain(1.25 / 1.5 / 2)` | +0.14* / +0.22* / −0.15 | −0.020 / −0.005 / −0.120* | 0.000 / +0.006* / +0.037* | −0.024* / −0.059* / −0.203* |
| Block 8–11 | `partial(0.25 / 0.5 / 0.75 / 1)` | +0.25* / +0.41* / +0.50* / +0.50* | +0.035* / +0.040 / +0.055 / +0.005 | −0.015* / −0.019* / −0.014* / −0.004 | +0.021* / +0.023 / +0.012 / −0.011 |
| Block 8–11 | `gain(1.25 / 1.5 / 2)` | +0.15* / +0.17 / −0.17 | +0.030 / −0.010 / −0.085* | 0.000 / +0.011* / +0.056* | +0.030* / +0.033 / −0.020 |

The best dose by REASON log-probability is `partial(0.75)` for both targets.

## Task 4: interchangeability (SEARCH, 50 TEXT chunks)

Results are in `task4/swap.csv`, with the heat map `task4/swap_heat.png`. The base loss on these
chunks is 2.52, and 288 swaps were run.

- **Typical cost:** the median loss increase is 0.13 to 0.21 at every distance from 1 to 8.
- **Cheap swaps:** 42 swaps cost less than 0.05 nats. All of them have i between 12 and 18 and j
  between 7 and 19, and 34 of the 42 run an earlier layer (j < i). The lowest is +0.005, for
  layer 7 run at position 13.
- **Expensive swaps:** 29 swaps cost more than 1 nat. All of them run layer 1, 3 or 24 to 27.

## Task 4b and 4c: do interchangeable layers share weights?

Results are in `task4b/pairs.csv` and `summary.json`, and `task4c/pairs.csv` and `summary.json`.
Plots are `task4b/match_vs_swap_*.png`, `task4b/best_match_hist.png` and
`task4c/match_vs_swap_*.png`. The note on the task 4b null is `task4b/NOTE.md`.

The 16 outlier coordinates have mean absolute values up to 63.2, against a median of 0.49.

**Positive control** (layer 14, reordered, noise at 30% of each matrix's standard deviation):
100% of neurons and 100% of heads are recovered, with and without the outlier set. It passes.

**Means by distance.** "4b" is the shuffled-coordinate null; "4c" is the covariance-matched null
on raw vectors. The mean-removed 4c variant is within 0.02 of the raw one everywhere. Neuron
cells give all coordinates / outliers removed.

| Distance | Neuron match, 4b | Neuron match, 4c | Mutual best match: real / 4c null | Median best similarity: real / 4c null | Head match, 4b | Head match, 4c |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.995 / 0.991 | 0.950 / 0.944 | 0.30 / 0.36 | 0.341 / 0.149 | 0.99 | 1.00 |
| 4 | 0.908 / 0.870 | 0.786 / 0.801 | 0.29 / 0.38 | 0.245 / 0.140 | 0.98 | 0.99 |
| 8 | 0.596 / 0.545 | 0.430 / 0.429 | 0.31 / 0.41 | 0.161 / 0.132 | 0.93 | 0.97 |

Mean matched head cosines at distance 1 are 0.054 (OV) and 0.072 (QK), against 0.000 for the
task 4b null.

**Link to task 4.** The cell gives the mean within-distance Spearman correlation with the swap
loss increase, then the one-sided permutation p for a negative mean.

| Measure | 4b, all | 4b, no outliers | 4c raw, all | 4c raw, no outliers | 4c mean-removed, all | 4c mean-removed, no outliers |
| --- | --- | --- | --- | --- | --- | --- |
| Neuron match fraction | −0.72, < 0.001 | −0.74, < 0.001 | −0.71, < 0.001 | −0.71, < 0.001 | −0.70, < 0.001 | −0.71, < 0.001 |
| Neuron mutual fraction | – | – | −0.60, < 0.001 | −0.56, < 0.001 | −0.58, < 0.001 | −0.54, < 0.001 |
| Head match fraction | −0.14, 0.011 | −0.16, 0.007 | −0.24, 0.001 | −0.24, < 0.001 | −0.24, < 0.001 | −0.24, < 0.001 |
| Head mutual fraction | – | – | −0.47, < 0.001 | −0.46, < 0.001 | −0.47, < 0.001 | −0.46, < 0.001 |

## Task 5: confirmation on REPORT (7B)

Results are in `task5/report.csv`. The REPORT base is: TEXT 2.615; REASON log-probability −2.92,
accuracy 0.39; RECALL accuracy 0.985; RECALL-HARD accuracy 0.904.

| Configuration | REASON log-prob | REASON accuracy | TEXT loss | RECALL log-prob | RECALL-HARD log-prob |
| --- | --- | --- | --- | --- | --- |
| Layer 14 `repeat` (also the mid-depth reference) | +0.62* | −0.075* | +0.023* | −0.011 | −0.097* |
| Layer 14 `gain(2)` (also the mid-depth reference) | +0.09 | −0.145* | +0.033* | +0.015 | −0.203* |
| Layer 14 `attn2` | −0.07 | +0.030 | +0.024* | −0.013* | +0.010 |
| Layer 14 `mlp2` | +0.46* | −0.090* | +0.018* | +0.005 | −0.076* |
| Layer 14 `partial(0.75)` | +0.57* | −0.030 | +0.013* | −0.010 | −0.079* |
| Block 8–11 `repeat` | +0.67* | 0.000 | −0.006 | +0.027* | −0.067 |
| Block 8–11 `gain(2)` | +0.08 | −0.125* | +0.055* | +0.026* | −0.170* |
| Block 8–11 `partial(0.75)` | +0.61* | −0.005 | −0.016* | +0.026* | +0.005 |
| Block 8–11 `partial(0.25)` | +0.27* | −0.010 | −0.015* | +0.012* | +0.033 |
| Layer: `repeat` − `gain(2)` | +0.53* | +0.070* | −0.010* | −0.026* | +0.107* |
| Block: `repeat` − `gain(2)` | +0.59* | +0.125* | −0.061* | +0.002 | +0.102 |
| Layer: `attn2` − `mlp2` | −0.53* | +0.120* | +0.006 | −0.018 | +0.085 |

## Task 5b: carry-over to 14B (REPORT)

Results are in `task5b/confirm.csv`.

- **Mapping:** layer 14 of 28 becomes layer 24 of 48. Block 8–11 becomes 14–20 (start 13.7
  rounded to 14, length 6.9 rounded to 7). The mid-depth layer is 24.
- **14B REPORT base:** TEXT 2.372; REASON log-probability −1.91, accuracy 0.47; RECALL accuracy
  1.00; RECALL-HARD accuracy 0.983.

| Configuration | REASON log-prob | REASON accuracy | TEXT loss | RECALL log-prob | RECALL-HARD log-prob |
| --- | --- | --- | --- | --- | --- |
| Layer 24 `repeat` (also the mid-depth reference) | −0.24* | −0.055 | +0.023* | −0.025* | +0.013 |
| Layer 24 `gain(2)` (also the mid-depth reference) | −0.47* | −0.025 | +0.026* | +0.003* | +0.010 |
| Layer 24 `attn2` | −0.09 | −0.040 | +0.006* | +0.005* | 0.000 |
| Layer 24 `mlp2` | −0.27* | −0.055* | +0.018* | −0.020* | +0.016 |
| Layer 24 `partial(0.75)` | −0.14* | −0.050 | +0.012* | −0.020* | +0.013 |
| Block 14–20 `repeat` | −0.30* | −0.040 | +0.018* | −0.028* | −0.041* |
| Block 14–20 `gain(2)` | −0.28* | −0.055 | +0.065* | −0.017 | −0.081 |
| Block 14–20 `partial(0.75)` | −0.15* | −0.035 | +0.003 | −0.015* | −0.024 |
| Block 14–20 `partial(0.25)` | −0.02 | 0.000 | −0.010* | −0.001 | −0.001 |
| Layer: `repeat` − `gain(2)` | +0.23* | −0.030 | −0.003 | −0.028* | +0.003 |
| Block: `repeat` − `gain(2)` | −0.02 | +0.015 | −0.047* | −0.011* | +0.039 |
| Layer: `attn2` − `mlp2` | +0.18 | +0.015 | −0.012* | +0.026* | −0.016 |

## Decision table

All of this table's rows refer to REPORT (tasks 5 and 5b) or to tasks 4, 4b and 4c.

| Result | Applies? | Basis |
| --- | --- | --- |
| No `repeat` configuration beats `base` with an interval excluding zero (QA no) | **No** on 7B | Layer 14 `repeat` +0.62* and block 8–11 `repeat` +0.67* on REASON log-probability. No configuration raises REASON accuracy (best is +0.030, not significant) |
| `repeat` beats `base`, and `repeat` − `gain(2)` excludes zero in favour of `repeat` (QB: thinking again) | **Yes** on 7B | Layer: +0.53* log-prob, +0.070* accuracy. Block: +0.59* log-prob, +0.125* accuracy |
| `repeat` beats `base`, but `gain(2)` does as well or better (QB: volume) | No | `gain(2)` is +0.09 (layer) and +0.08 (block), neither excluding zero |
| `attn2` − `mlp2` excludes zero in favour of `attn2` (QC) | **On accuracy only** | Accuracy +0.120* in favour of `attn2`; log-probability −0.53* in favour of `mlp2`. The table does not say which score decides |
| `mlp2` as good as or better than `attn2` | **On log-probability only** | See the row above |
| Gain on REASON with no gain or a loss on RECALL-HARD (QD) | **Yes** | Every configuration that raises REASON log-probability leaves RECALL-HARD unchanged (block configurations) or lowers it (layer 14: −0.076* to −0.097*) |
| Similar change on REASON and RECALL-HARD | No | |
| Any gain on REASON comes with a loss on TEXT | **Layer 14: yes; block 8–11: no** | Layer 14: TEXT +0.013* to +0.023*. Block 8–11: `repeat` −0.006 (no significant change); `partial(0.75)` −0.016* and `partial(0.25)` −0.015* (TEXT improves) |
| Swap heat map shows bands of low loss increase (QE) | **Yes, one region** | All 42 swaps under 0.05 nats have i between 12 and 18 and j between 7 and 19 (34 of them with j < i) |
| A task 5 effect that counts on 7B has the same sign and an interval excluding zero on 14B | **For some effects** | Carries over: layer `repeat` − `gain(2)` on REASON log-prob (+0.53* → +0.23*); TEXT loss rise of layer `repeat`, `gain(2)`, `mlp2` and `partial(0.75)`; TEXT fall of block `partial(0.25)` (−0.015* → −0.010*); block `repeat` − `gain(2)` on TEXT (−0.061* → −0.047*) |
| The same effect is absent or reversed on 14B | **Yes for every REASON gain over `base`** | Reversed: layer `repeat` (+0.62* → −0.24*), layer `mlp2` (+0.46* → −0.27*), layer `partial(0.75)` (+0.57* → −0.14*), block `repeat` (+0.67* → −0.30*), block `partial(0.75)` (+0.61* → −0.15*). Absent: block `partial(0.25)` (+0.27* → −0.02); block `repeat` − `gain(2)` on REASON (+0.59* → −0.02) |
| Task 4b positive control failed | No | 100% of neurons and heads recovered |
| Task 4c: neuron match fraction against the covariance null above 0.2 at distance 1, and within-distance correlation p below 0.05 | **Yes** | 0.950 (0.944 without outliers); correlation −0.70 to −0.71, p < 0.001. Mutual best matches are rarer for real pairs than against the null (0.30 against 0.36) |
| Task 4c: match fractions near 0.01 against the covariance null, but the correlation survives | No | |
| Task 4c: correlation gone (p above 0.05) | No | |
| Neuron or head match fraction above 0.2 for neighbouring layers, outlier set removed (QF) | **Yes** | 4b: 0.991 neurons, 0.99 heads; 4c: 0.944, 1.00 |
| Match fractions near the null level (below 0.05) at every distance, outlier set removed | No | |
| Matches above 0.05 only with the outlier set kept | No | |
| Mean within-distance correlation between match fraction and swap loss increase negative, permutation p below 0.05 (QE and QF agree) | **Yes** | Neurons: −0.72 / −0.74 (4b), −0.70 to −0.71 (4c). Heads: −0.14 / −0.16 (4b), −0.24 (4c). All p ≤ 0.011 |
| Correlation near zero or p above 0.05 | No | |

**Stopped for review.**
