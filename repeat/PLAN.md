# Experiment C: repeating layers of a real model, with finer tools

A third, independent experiment. It takes an existing open language model, changes how data flows
through its layers without changing any weight, and measures what happens. No training.

Background: the "Repeat Yourself" posts (dnhkng.github.io/posts/rys-ii) report that running a
block of middle layers twice improves a large model. That intervention is blunt: it repeats
attention and MLP together, for every token, at full strength, and it was selected on very small
question sets. This experiment asks what exactly helps, if anything.

The rules in `CLAUDE.md` apply. In addition:

- All code lives in `repeat/`, all outputs in `results/repeat/`. Do not edit any file outside
  `repeat/` and `results/repeat/`.
- Model weights go to `data/cache/hf/` (git-ignored). Check free disk space before downloading.
- Do not start a run while another run is using the GPU. Check with `ps` first.
- Commit messages: `repeat task N: <what>`.

## 1. Questions

| | Question | Compared |
| --- | --- | --- |
| QA | Does repeating any layer or block help at all, on questions it was not selected on? | repeat vs unmodified |
| QB | Is a second pass "thinking again", or only a larger step in the same direction? | repeat vs doubling the update without recomputing |
| QC | Which half of a layer matters? | attention-only repeat vs MLP-only repeat |
| QD | Does repetition help reasoning more than recall? | gain on multi-step questions vs gain on single facts |
| QE | Which layers can stand in for each other? | layer j run in place of layer i |
| QF | Do layers that can stand in for each other also have matching weights, once the order of their neurons and heads is ignored? | aligned weights of layer i and layer j, against a shuffled null and against QE |

## 2. Notation and interventions

A layer maps `x` to `y` in two parts: `x1 = x + A(x)` (attention) and `y = x1 + M(x1)` (MLP).
Write `L(x) = y`, and `u = y - x` for the layer's update. All interventions act on one layer or
one contiguous block of layers; for a block, `L` is the whole block and `u` its total update.

| Name | Output in place of `y` | Meaning |
| --- | --- | --- |
| `base` | `y` | Unmodified |
| `repeat` | `L(y)` | Second pass, recomputed (the Repeat Yourself intervention) |
| `gain(g)` | `x + g*u` | Same update scaled, nothing recomputed. `gain(2)` is the control for `repeat`; `gain(0)` skips the layer |
| `partial(a)` | `y + a*(L(y) - y)` | A fraction of the second pass. `partial(1)` equals `repeat` |
| `attn2` | `y + A(y)` | Second pass of attention only (single layers only) |
| `mlp2` | `y + M(y)` | Second pass of the MLP only (single layers only) |
| `swap(i,j)` | layer j's weights run at position i | Interchangeability |

## 3. Model

Main model: `Qwen/Qwen2.5-7B-Instruct` (28 layers, about 15 GB in float16) on MPS, used as a
plain text predictor with no chat template. It runs about twice as fast as the 14B model, which
matters because tasks 1 to 4 evaluate hundreds of configurations. Development model:
`Qwen/Qwen2.5-0.5B-Instruct`. Write and debug everything on the development model; its numbers
are not results.

Confirmation model: `Qwen/Qwen2.5-14B-Instruct` (48 layers), used only in Repeat task 5b to check
whether the 7B findings carry over to a deeper model. Layer ranges in this plan are given as
fractions of depth so that a configuration can be moved from one model to the other.

## 4. Measurements

Every configuration is scored on four sets with one teacher-forced forward pass per item.

| Set | Items | Score |
| --- | --- | --- |
| TEXT | 200 chunks of 256 tokens from the validation split of `wikitext-103-raw-v1` | Mean loss per token (lower is better) |
| REASON | 400 generated multi-step arithmetic questions | Mean log-probability of the correct answer; accuracy (every answer token is the top prediction) |
| RECALL | 300 single-fact questions: about 200 country capitals and 100 chemical element symbols | Same two scores |
| RECALL-HARD | 236 single-fact questions: the atomic number of each of the 118 elements, and the element with each atomic number (added by Repeat task 0b) | Same two scores |

REASON item format, with three fixed worked examples in front and k steps on small integers:

```
Start with 7. Add 5. Multiply by 3. Subtract 4. The result is 32.
```

The answer is the number after "The result is". Operations: add, subtract, multiply by 2 to 5;
all intermediate values between 0 and 999. RECALL format, with three fixed examples in front:
`The capital of France is Paris.` and `The chemical symbol for gold is Au.`
RECALL-HARD format, with three fixed examples in front (elements excluded from the items):
`The atomic number of gold is 79.` and `The element with atomic number 79 is gold.`

**Splits.** Each set is split in half with seed 0 into SEARCH and REPORT. Every choice of
configuration is made on SEARCH. REPORT is touched only in Repeat task 5. This is the guard
against finding a lucky configuration.

**Statistics.** Every reported difference is paired (same items, two configurations) with a 95%
bootstrap interval over items, 1,000 resamples, seed 0. A difference "counts" only if its
interval excludes zero.

## 5. Tasks for the agent

### Repeat task 0: code, exactness check, speed, difficulty

1. Write `repeat/flow.py`: a forward pass that walks the model's layers itself, so each
   intervention in section 2 can be applied. Unit checks on the development model, all must pass:
   - With no intervention, logits equal the library's own forward pass (max difference under
     1e-3 in float32).
   - `gain(1)` and `partial(0)` equal `base`. `partial(1)` equals `repeat`. `swap(i,i)` equals
     `base`.
2. Write `repeat/data.py` (the three sets and the splits) and `repeat/evaluate.py`.
3. Load the main model, repeat the exactness check in float16 (tolerance 5e-2 on logits), and
   measure tokens per second on the SEARCH half of all three sets.
4. **Difficulty of REASON.** The unmodified model's accuracy on SEARCH must be between 30% and
   70% so that both gains and losses are visible. Try k = 3 steps; if accuracy is above 70% try
   k = 4, then 5, then 6; if below 30% at k = 3 try k = 2. Freeze the first k in range. If none
   is in range, write `STOP.md`.
5. Record baseline scores on SEARCH for all three sets. Do not look at REPORT.

Stop for review. Report speed, chosen model, chosen k, baseline scores.

### Repeat task 0b: a recall set below the ceiling (added after task 0)

Task 0 found RECALL at 100% accuracy on SEARCH, so accuracy cannot rise and the reasoning versus
recall comparison (QD) would rest on log-probability alone. RECALL stays as it is. Add
RECALL-HARD (section 4) to `repeat/data.py`, split it into SEARCH and REPORT with seed 0 like the
others, and record the unmodified model's SEARCH scores. Use the standard element names
(IUPAC English spelling, "aluminium" and "caesium" accepted as written in your table; pick one
spelling per element and keep it).

Report its baseline accuracy. If it is above 0.9, also report it and continue: do not change the
set again. From task 1 onward every configuration is scored on all four sets. QD is judged on
RECALL-HARD; RECALL is reported next to it.

### Repeat task 1: single-layer map (SEARCH)

For every layer: `repeat`, `gain(2)`, `gain(0)`, `attn2`, `mlp2`. Five configurations per layer,
all four sets. Output `results/repeat/task1/layers.csv` and one plot per set of score change
against layer for the five interventions.

### Repeat task 2: blocks (SEARCH)

Contiguous blocks of length 2, 4 and 8, starting at every second layer between 20% and 80% of
depth: `repeat` and `gain(2)`. Output `results/repeat/task2/blocks.csv` and a heat map of REASON
log-probability change (start layer by length) for each of the two interventions.

### Repeat task 3: dose (SEARCH)

Take the single layer and the block with the largest REASON log-probability gain under `repeat`
in tasks 1 and 2 (if nothing has a positive gain, take the least negative and say so). For each:
`partial(a)` at a = 0.25, 0.5, 0.75, 1 and `gain(g)` at g = 1.25, 1.5, 2. Output
`results/repeat/task3/dose.csv` and one plot.

### Repeat task 4: interchangeability (SEARCH, TEXT only)

For layers i between 15% and 85% of depth and every j with |i - j| at most 8: `swap(i,j)`, scored
on 50 TEXT chunks. Output `results/repeat/task4/swap.csv` and a heat map of loss increase.

### Repeat task 4b: do interchangeable layers share weights? (weights only)

Task 4 measures interchangeability by behaviour. This task compares the weights themselves. It
runs no evaluation set except one short measurement in step 1, so it uses neither SEARCH nor
REPORT items for selection.

**What may and may not differ between two layers.** All layers read from and write to the same
residual stream, so the 3,584 residual coordinates (896 on the development model) mean the same
thing in every layer, and no alignment is needed on that side. What is arbitrary is the order of
the hidden units inside a layer: the MLP neurons, the attention heads, and the KV groups. Inside
a head, only the products `W_Q W_K^T` and `W_V W_O` are free of internal basis choices. So the
comparison matches neurons to neurons and heads to heads, and compares vectors in residual
coordinates.

Load weights one layer at a time from the safetensors files on CPU. Compute in float32, on MPS
in chunks for the large products. Never hold more than two layers in memory.

1. **Preparation.**
   - Fold each RMSNorm gain into the weights that read from it: multiply the columns of
     `q_proj`, `k_proj`, `v_proj` by the input norm's gain, and those of `gate_proj` and
     `up_proj` by the post-attention norm's gain.
   - **Outlier dimensions.** Run the unmodified model on the first 20 TEXT SEARCH chunks and
     record the mean absolute value of each residual coordinate at the input of the layer at 50%
     of depth. The 16 largest are the outlier set `O`. Every measurement below is done twice:
     on all coordinates, and with the coordinates in `O` set to zero in every vector. A few huge
     shared coordinates can make unrelated vectors look alike; the second version removes that.
2. **MLP neuron vectors.** Neuron n of a layer has three vectors in residual coordinates: its
   gate row `g_n`, its up row `u_n` (rows of `gate_proj` and `up_proj`), and its output column
   `d_n` (column of `down_proj`). The similarity of neuron n in layer i and neuron m in layer j is
   `s(n,m) = cos(g_n, g_m) + |cos(u_n, u_m) + cos(d_n, d_m)|`, between -1 and 3. The absolute
   value is there because flipping the signs of both `u` and `d` leaves the neuron unchanged.
   For each neuron of layer i, record its best match in layer j (the m with the largest s), and
   whether the match is mutual (n is also m's best match in layer i).
3. **Attention heads.** For Q head h, with `W_V` from its KV group:
   - `OV_h = W_V^T W_O,h`, a 3,584 by 3,584 matrix of rank 128. Never form it: the cosine
     between two such products, `<A, B> / (|A| |B|)` with `<A, B> = trace((U_A^T U_B)(V_B V_A^T))`
     for `A = U_A V_A`, needs only 128 by 128 products.
   - `QK_h = W_Q,h^T W_K` in the same way. Rotary position embedding means this is the attention
     pattern at relative offset 0 only; say so in the report.
   Head similarity = mean of the OV cosine and the QK cosine. Match the heads of layer i to those
   of layer j one to one with `scipy.optimize.linear_sum_assignment`, maximising total similarity.
4. **Null.** Build a shuffled copy of layer j: apply an independent random permutation of the
   residual coordinates to every vector (each `g_m`, `u_m`, `d_m`, and each head's factor
   matrices, seed 0). It keeps every value and every norm but breaks the alignment with the
   residual stream. Recompute steps 2 and 3 against it. A neuron of layer i "has a match" if its
   best similarity in the real layer j is above the 99th percentile of its best similarity
   against the null. Same rule for matched heads.
5. **Positive control.** Make a copy of one layer at 50% of depth with its neurons and heads
   randomly reordered, and Gaussian noise added to every weight at 30% of that weight matrix's
   standard deviation. Compare it with the original. At least 90% of neurons and heads must be
   recovered (best match = the true partner). If not, the method cannot find real matches; write
   `STOP.md`. Do this on the development model first, then on the main model.
6. **Pairs.** Exactly the (i, j) pairs of task 4: i between 15% and 85% of depth, j with
   |i - j| at most 8, j different from i. Similarities are symmetric in the summary but the
   best-match fraction is not, so compute both directions.
7. **Per pair, record:** fraction of neurons of i with a match in j, fraction of mutual matches,
   median best-match similarity, the same median against the null, fraction of heads with a
   match, mean matched OV cosine, mean matched QK cosine, and the same two means against the
   null. Each with and without the outlier set.
8. **Link to task 4.** Across all pairs, the Spearman correlation between the fraction of
   neurons with a match and the task 4 loss increase of `swap(i,j)`; the same for the fraction
   of heads with a match. Distance alone drives both (neighbouring layers are more alike in both
   respects), so also compute it within each distance |i - j| = 1 to 8, and report the mean of
   the eight within-distance correlations. Its p-value comes from a permutation test: within each
   distance, shuffle the similarity values across pairs 1,000 times (seed 0) and recompute the
   mean.
9. **Groups of layers.** From the pair table, build a layer-by-layer matrix of neuron match
   fraction (mean of the two directions; pairs further than 8 apart left empty). Plot it next
   to the task 4 heat map, on the same axes.

Output: `results/repeat/task4b/pairs.csv` (one row per ordered pair and outlier setting),
`results/repeat/task4b/summary.json` (positive control, correlations, p-values), heat maps of
neuron match fraction and head match fraction next to the task 4 swap map, and a histogram of
best-match similarity for real and null at |i - j| = 1, 4 and 8.

Expected cost on the main model: about 150 unordered pairs, each a few large matrix products,
so roughly 30 to 60 minutes, mostly reading weights from disk. Weight reading dominates: read
each layer once per neighbourhood and reuse it, rather than once per pair.

### Repeat task 5: confirm on REPORT

Evaluate on the REPORT halves, with paired bootstrap intervals against `base`:

- the best single layer and the best block from task 3 under `repeat`;
- for the same layer and block: `gain(2)`, and for the layer `attn2` and `mlp2`;
- the best dose found in task 3 for each;
- added after task 3, before any REPORT item was seen: block 8–11 under `partial(0.25)`, the
  only SEARCH setting where REASON, TEXT and RECALL-HARD all improved significantly;
- as a selection-free reference, `repeat` and `gain(2)` on the single layer at 50% of depth.

Also the paired difference `repeat` minus `gain(2)` for the layer and for the block (QB), and
`attn2` minus `mlp2` (QC).

### Repeat task 4c: a fair null for the weight comparison (added after task 4b)

Task 4b's shuffled-coordinate null destroys the structure that all vectors of a layer share (a
common mean direction, the outlier coordinates, a dominant low-rank part), so almost any real
neuron beats it: match fractions were 0.99 at distance 1 and still 0.55–0.60 at distance 8.
Weights only, reusing task 4b's code; do not change task 4b's outputs.

1. **Covariance-matched null.** For each layer j and each vector type (`g`, `u`, `d`, after
   folding the norm gains), compute the mean and covariance of its vectors over all neurons of
   that layer in float64 on CPU. Draw as many null vectors as the layer has neurons from that
   Gaussian (seed 0). A null "neuron" takes one draw of each type. For heads, draw each factor
   matrix's rows the same way from the layer's own row distribution. This keeps what a whole
   layer has in common and breaks only the identity of individual units.
2. **Mean removed.** Repeat task 4b's real comparison after subtracting each layer's mean vector
   (per type) from its neurons' vectors, so shared direction cannot create matches.
3. **Headline measure.** Per pair, report: the fraction of neurons with a match against the new
   null (99th percentile rule, as before), the fraction of mutual best matches, and the same
   mutual fraction for layer i against the null layer. Same for heads.
4. **Link to task 4 again.** Recompute the mean within-distance Spearman correlation and its
   permutation p-value with the new match fraction, and with the mutual fraction.

Output: `results/repeat/task4c/pairs.csv`, `summary.json`, and the two heat maps redrawn.
About 30 minutes.

### Repeat task 5b: carry over to the 14B model (REPORT)

Load `Qwen/Qwen2.5-14B-Instruct` once, after task 5 is committed. Rerun the exactness check from
task 0 on it (float16, tolerance 5e-2). Then evaluate on the REPORT halves, with the same paired
bootstrap against its own `base`:

- the task 5 configurations, moved to the 14B model by fraction of depth: layer i of 28 becomes
  the 14B layer nearest to i × 48 / 28; a block keeps its start fraction and its length fraction,
  rounded to whole layers;
- the selection-free reference: `repeat` and `gain(2)` on the layer at 50% of depth.

No search is done on the 14B model, so nothing about it is selected on its own data. Output
`results/repeat/task5b/confirm.csv`. If the 14B model does not load or runs under 300 tokens per
second, skip this task and say so in the report.

### Repeat task 6: report

Write `results/repeat/REPORT.md` with the tables and plots. No interpretation beyond marking
which rows apply:

| Result on REPORT | What it means |
| --- | --- |
| No `repeat` configuration beats `base` with an interval excluding zero | The reported effect does not reproduce on this model and these questions (QA no) |
| `repeat` beats `base`, and `repeat` minus `gain(2)` excludes zero in favour of `repeat` | The second pass computes something new (QB: thinking again) |
| `repeat` beats `base`, but `gain(2)` does as well or better | The benefit is a larger step, not recomputation (QB: volume) |
| `attn2` minus `mlp2` excludes zero in favour of `attn2` | Repeating routing matters, repeating lookup does not (QC) |
| `mlp2` as good as or better than `attn2` | The facts-versus-procedure reading of the two halves is not supported here |
| Gain on REASON with no gain or a loss on RECALL-HARD | Repetition helps multi-step work specifically (QD) |
| Similar change on REASON and RECALL-HARD | No evidence it is specific to reasoning |
| Any gain on REASON comes with a loss on TEXT | It trades general quality for the task |
| Swap heat map shows bands of low loss increase | Groups of layers are interchangeable (QE) |
| A task 5 effect that counts on 7B has the same sign and an interval excluding zero on 14B (task 5b) | The effect carries over to a deeper model |
| The same effect is absent or reversed on 14B | The effect is specific to the 7B model |
| Task 4b positive control failed | The weight comparison cannot find matches; no row below applies |
| Task 4c: neuron match fraction against the covariance null above 0.2 at distance 1, and the within-distance correlation with swap cost still has p below 0.05 | Neighbouring layers share individual units beyond what any layer shares, and that sharing predicts interchangeability |
| Task 4c: match fractions near 0.01 against the covariance null, but the correlation survives | Interchangeable layers are alike as wholes (shared directions), not unit by unit |
| Task 4c: correlation gone (p above 0.05) | Task 4b's link to swap cost came from layer-wide structure the old null missed |
| Neuron or head match fraction above 0.2 for neighbouring layers with the outlier set removed | Neighbouring layers share many neurons or heads in weight space (QF) |
| Match fractions near the null level (below 0.05) at every distance, outlier set removed | Interchangeable or not, layers do not share recognisable neurons or heads (QF no) |
| Matches above 0.05 only with the outlier set kept | Apparent sharing comes from a few large residual coordinates, not from shared units |
| Mean within-distance correlation between match fraction and swap loss increase is negative, permutation p below 0.05 | Layers that can stand in for each other are the ones whose weights match (QE and QF agree) |
| Correlation near zero or p above 0.05 | Interchangeability is not explained by shared units |

Stop for review.

## 6. Budget

Unknown until task 0. Rough count: task 1 is 5 configurations per layer, task 2 about 60 to 70,
task 4 about 300 on a small set, task 4b 20 to 40 minutes of weight comparison, task 5b about
10 configurations on the 14B model. At a guessed 12 seconds per full configuration on the 7B
model this is 1.5 to 3 hours. If task 0 measures more than 60 seconds per configuration, halve the
SEARCH items for tasks 1, 2 and 4 and say so.

## 7. Limits

- One model family. Search is done on 7B only; 14B only confirms. The original reports used
  larger models with more layers; a null result on smaller ones does not refute them.
- The model was never trained to be run this way. This measures slack in an existing model, not
  what a model trained with a loop could do (that is Experiment B).
- REASON is arithmetic only. A gain there is not a gain in "reasoning" in general.
- Interventions apply to every token position. Selecting positions is a possible follow-up.
- Task 4b compares units one to one. Two layers can compute the same function with different
  neurons in superposition, which a one-to-one comparison misses; matching subspaces (for
  example with an SVD of the stacked weights, as in Basis Sharing, arXiv 2410.03765) is the
  next step if units do not match. It also ignores rotations inside a head beyond the two
  invariant products.
