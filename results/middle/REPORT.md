# Experiment E report: what the middle layers compute

Model: Qwen2.5-7B-Instruct, float16 on MPS. Seed 0. Plan: `middle/PLAN.md`. Every script was
first run on Qwen2.5-0.5B-Instruct; those outputs are in the `dev/` subfolders and are not
results. Zones: ENTRY 0–5, FAMILY 6–17, OUTPUT 18–27. Outlier coordinates: the 16 from Repeat
task 4b. Numbers are as measured; the only interpretation is the decision table at the end.

## Part 1: layer operators

### Task 1: statistics

Data and setup:

- 480 chunks of 256 tokens from wikitext-103-raw-v1 **train** (124.2M tokens, 485,332 chunks
  available), chosen with seed 0 and kept in corpus order.
- Positions 8–255 only.
- FIT is chunks 0–379, with 94,240 positions. It is stored as two halves, FIT_A (0–189) and
  FIT_B (190–379), for the noise ceiling.
- TEST is chunks 380–479, with 24,800 positions.

The sums were accumulated in 4 passes of up to 8 layers, at 3.3k–6.2k tokens/s, in 132 s. All
checks in `task1/metrics.json` pass:

- The layer inputs match the library's hidden states exactly (relative error 0).
- The constant coordinate's sum equals n exactly, and every Szz is exactly symmetric.
- Mean z² is between 0.9957 and 1.0000.

### Task 2: one operator per layer

`task2/layers.csv`, `task2/r2_per_layer.png`. TEST R² of each layer's own ridge map (R²_ll),
next to the half-on-half noise ceiling:

| Zone | R²_ll mean (all) | min | ceiling mean | R²_ll mean (no outliers) | ceiling |
| --- | --- | --- | --- | --- | --- |
| ENTRY | 0.393 | 0.293 | 0.372 | 0.375 | 0.353 |
| FAMILY | 0.361 | 0.325 | 0.329 | 0.347 | 0.314 |
| OUTPUT | 0.395 | 0.255 | 0.368 | 0.373 | 0.345 |

The family layers, 6 to 17 in order, with all coordinates: 0.359, 0.355, 0.337, 0.399, 0.404,
0.329, 0.360, 0.347, 0.397, 0.386, 0.325, 0.334. The highest layer is 27 (0.812).

The ridge penalty c chosen by TEST R² is 0.1, the largest value in the plan's grid, for 16 of
the 28 layers (1–9 and 20–26), and 0.01 for the other 12. This holds in both settings.

### Task 3: does one layer's operation explain another's

`task3/pairs.csv` has all 28 × 28 pairs in both settings. Heat maps are in
`task3/heatmaps_all.png` and `task3/heatmaps_no_outliers.png`: the ratio, read overlap and write
overlap, each next to its null, with colours clipped to [0, 1].

Means over pairs i ≠ j within a zone (ratio = R²_ij,scaled / R²_ii; a = best scale):

| Setting | Zone | ratio | null | ratio, pairs ≤ 4 apart | null, ≤ 4 apart | min pair ratio | mean a | read overlap (null) | write overlap (null) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | FAMILY | −0.084 | −0.126 | −0.078 | −0.128 | −0.224 | 0.187 | 0.189 (0.113) | 0.189 (0.113) |
| all | OUTPUT | −0.392 | −0.466 | −0.366 | −0.468 | −0.900 | 0.130 | 0.154 (0.113) | 0.158 (0.114) |
| all | ENTRY | −0.370 | −0.395 | −0.359 | −0.386 | −0.743 | 0.100 | 0.150 (0.114) | 0.184 (0.113) |
| no outliers | FAMILY | −0.055 | −0.093 | −0.051 | −0.093 | −0.138 | 0.181 | 0.188 (0.113) | 0.179 (0.114) |
| no outliers | OUTPUT | −0.214 | −0.226 | −0.189 | −0.207 | −0.603 | 0.075 | 0.152 (0.113) | 0.145 (0.114) |
| no outliers | ENTRY | −0.211 | −0.224 | −0.210 | −0.224 | −0.340 | 0.081 | 0.149 (0.114) | 0.173 (0.114) |

Inside the family (all coordinates), the ratio, overlaps and null by distance between the layers:

| \|i − j\| | 1 | 2 | 3 | 4 | 6 | 8 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ratio | −0.101 | −0.061 | −0.075 | −0.072 | −0.093 | −0.099 | −0.103 |
| null ratio | −0.126 | −0.131 | −0.126 | −0.127 | −0.127 | −0.126 | −0.119 |
| read overlap | 0.245 | 0.224 | 0.191 | 0.179 | 0.163 | 0.143 | 0.139 |
| write overlap | 0.265 | 0.227 | 0.176 | 0.168 | 0.156 | 0.148 | 0.160 |

Across the 132 ordered family pairs, the largest ratio is 0.010, and 3% of pairs have a ratio
above 0.

The plan's R² removes the mean of u from `tot`, so a map that predicts zero scores
R² = −n|ū|²/tot. That is how values below 0 arise. Computed from the TEST sums, this zero-map R²
ranges from −0.022 to −0.089 in the family and from −0.070 to −0.509 in OUTPUT (the most negative
is layer 27). This is a description of the metric; the metric itself was not changed.

## Part 2: two-hop questions

### Task 4: data

`middle/countries.py` holds 88 countries. 102 others were dropped, each with the rule that
removed it (R1–R7: language, capital, status, continent, name, leak, answer name). The plan
aimed for about 120 rows; the rules were not relaxed to reach that number. France, Japan and
Argentina are used only as the three worked examples. That leaves 85 items. Greedy first-token
accuracy (`task4/metrics.json`, per item in `task4/items.json`):

| HOP1 | HOP2-language | HOP2-continent | TWO-language | TWO-continent |
| --- | --- | --- | --- | --- |
| 0.988 | 1.000 | 1.000 | 0.929 | 1.000 |

84 items are eligible for each kind. Among them, TWO-language solves 78 (accuracy 0.929) and
fails 6 (Senegal, Ghana, Zambia, Malawi, Botswana, Belize). TWO-continent solves all 84 and fails
none.

### Task 5: where each step happens

`task5/items.csv`, `task5/lens.png`, `task5/reuse.png`. Layers are counted as "after layer l"
(0–27). Top 10 means rank < 10.

| Kind | L_bridge median [IQR] (found) | L_answer1 median [IQR] | L_answer2 median [IQR] | delay median [IQR], mean | L_answer2 in OUTPUT |
| --- | --- | --- | --- | --- | --- |
| language | 24 [23, 24] (76 of 84) | 23 [22, 23] | 23 [22, 24] | 0 [0, 1], 0.43 | 84 of 84 |
| continent | 25 [21.75, 25] (8 of 84) | 21 [21, 22] | 21 [21, 22] | 0 [0, 0], 0.10 | 84 of 84 |

For the 6 failed language items, the delay has median 0.5 and mean 0.5.

Reuse: the cosine between the three-layer update sums of HOP2 (ending at L_answer1) and TWO
(ending at L_answer2), outlier coordinates removed, with bootstrap 95% CIs:

| Kind | real | null (different item, different answer) | real − null |
| --- | --- | --- | --- |
| language | 0.782 [0.742, 0.818] | 0.365 [0.332, 0.396] | 0.417 [0.375, 0.461] |
| continent | 0.865 [0.839, 0.887] | 0.698 [0.674, 0.724] | 0.167 [0.133, 0.203] |

Fix after the first run: the first version of the null gave almost every item the same partner
(Latvia), because it took the first suitable item of one shuffled order. It now puts the items
in a seeded random cycle and gives each item the next suitable item after it. Each partner is
now used at most 3 times. Both runs were redone (commit `8fb3ea9`). The real cosines are
unchanged; the language null mean changed from 0.451 to 0.365, and the continent null from 0.698
to 0.698.

### Task 6: back-patching

`task6/backpatch.csv` covers all 78 (i, j) pairs with i < j, both even, from 2 to 26.
`task6/backpatch.png` shows fixed and broken rates for both kinds, each next to its control.
Patching an item's own state at the same layer (j = i) reproduces task 4's result for every i,
as a check.

Only the 6 failed TWO-language items can be fixed; continent has no failed items. So every
"fixed" fraction is out of 6 items.

- **Cells with at least 20% fixed:** 40 of 78. The control (source taken from another item at
  the same layer) reaches at least 20% in 43 of 78.
- **Average over all cells:** 2.05 of 6 items fixed, against 1.90 for the control.
- **With i in FAMILY:** the mean fixed fraction is 0.352, against 0.348 for the control.
- **Largest gaps with i in FAMILY:** (i, j) = (10, 22) fixes 6 of 6, against 3 of 6 for the
  control. Next come (10, 24), (12, 24) and (14, 24), each 4 of 6 against 1 of 6.
- **Fixed for every i from 4 to 12 at j = 22:** all 6 items. The control fixes 3 to 5 of them in
  those cells.
- **Solved items broken, mean over cells (both kinds):** 0.030, against 0.293 for the control.

## Decision table

| Result | Mark | Numbers |
| --- | --- | --- |
| Family layers have own-map R² of at least 0.5 | **Does not apply** | Family R²_ll 0.325–0.404, mean 0.361 (0.347 without outliers) |
| Within the family, mean ratio ≥ 0.5 for pairs up to 4 apart, and rotated null ≤ 0.1 | **Does not apply** | Ratio −0.078 (null −0.128); without outliers −0.051 (−0.093) |
| The ratio stays at least 0.3 across the whole family | **Does not apply** | Smallest pair −0.224, largest 0.010 |
| Ratio inside the family clearly above OUTPUT's (by at least 0.2) | **Applies with all coordinates, not without outliers** | Family −0.084 vs OUTPUT −0.392 (difference 0.308); without outliers −0.055 vs −0.214 (difference 0.159). Both zones' values are below 0. |
| Read overlap high but write overlap low (or the reverse) within the family | **Does not apply** | Read 0.189, write 0.189; null 0.113 for both |
| Median delay at least 2 layers and L_answer2 often in OUTPUT | **Does not apply** | Median delay 0 for both kinds. L_answer2 is in OUTPUT for all 168 items (layers 21–26). |
| Reuse cosines clearly exceed the null | **Applies** | Real − null 0.417 [0.375, 0.461] for language and 0.167 [0.133, 0.203] for continent |
| Back-patching fixes ≥ 20% of failures for some (i, j) with i in FAMILY, and well above the control | **First part applies; "well above the control" is not shown** | Up to 6 of 6 fixed (e.g. (10, 22)), but the control fixes 3 of 6 there. The largest gap is 3 items. There are only 6 failed items. |
| Back-patching fixes no more than the control | **Applies on average over cells** | 2.05 vs 1.90 items of 6 over all cells; 0.352 vs 0.348 with i in FAMILY |

## Notes on the method

- The ridge penalty sits at the top of the plan's grid (c = 0.1) for 16 of 28 layers. The plan
  fixes the grid, so it was not extended.
- The country table has 88 rows instead of about 120.
- Back-patching rests on 6 failed items of one kind.
- One implementation fix was made after seeing results, to the pairing of the reuse null in task
  5, as described above. No metric, threshold or test set was changed.
