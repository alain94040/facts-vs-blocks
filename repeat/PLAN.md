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

Main model: `Qwen/Qwen2.5-14B-Instruct` (48 layers) in float16 on MPS, used as a plain text
predictor with no chat template. Development model: `Qwen/Qwen2.5-0.5B-Instruct`. Write and debug
everything on the development model; its numbers are not results.

Fallback, decided once in Repeat task 0: if the main model does not load, or runs the evaluation
below at under 500 tokens per second, use `Qwen/Qwen2.5-7B-Instruct` (28 layers) and say so in
the report. Layer ranges in this plan are given as fractions of depth for that reason.

## 4. Measurements

Every configuration is scored on three sets with one teacher-forced forward pass per item.

| Set | Items | Score |
| --- | --- | --- |
| TEXT | 200 chunks of 256 tokens from the validation split of `wikitext-103-raw-v1` | Mean loss per token (lower is better) |
| REASON | 400 generated multi-step arithmetic questions | Mean log-probability of the correct answer; accuracy (every answer token is the top prediction) |
| RECALL | 300 single-fact questions: about 200 country capitals and 100 chemical element symbols | Same two scores |

REASON item format, with three fixed worked examples in front and k steps on small integers:

```
Start with 7. Add 5. Multiply by 3. Subtract 4. The result is 32.
```

The answer is the number after "The result is". Operations: add, subtract, multiply by 2 to 5;
all intermediate values between 0 and 999. RECALL format, with three fixed examples in front:
`The capital of France is Paris.` and `The chemical symbol for gold is Au.`

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
   measure tokens per second on the SEARCH half of all three sets. Apply the fallback rule.
4. **Difficulty of REASON.** The unmodified model's accuracy on SEARCH must be between 30% and
   70% so that both gains and losses are visible. Try k = 3 steps; if accuracy is above 70% try
   k = 4, then 5, then 6; if below 30% at k = 3 try k = 2. Freeze the first k in range. If none
   is in range, write `STOP.md`.
5. Record baseline scores on SEARCH for all three sets. Do not look at REPORT.

Stop for review. Report speed, chosen model, chosen k, baseline scores.

### Repeat task 1: single-layer map (SEARCH)

For every layer: `repeat`, `gain(2)`, `gain(0)`, `attn2`, `mlp2`. Five configurations per layer,
all three sets. Output `results/repeat/task1/layers.csv` and one plot per set of score change
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

### Repeat task 5: confirm on REPORT

Evaluate on the REPORT halves, with paired bootstrap intervals against `base`:

- the best single layer and the best block from task 3 under `repeat`;
- for the same layer and block: `gain(2)`, and for the layer `attn2` and `mlp2`;
- the best dose found in task 3 for each;
- as a selection-free reference, `repeat` and `gain(2)` on the single layer at 50% of depth.

Also the paired difference `repeat` minus `gain(2)` for the layer and for the block (QB), and
`attn2` minus `mlp2` (QC).

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
| Gain on REASON with no gain or a loss on RECALL | Repetition helps multi-step work specifically (QD) |
| Similar change on REASON and RECALL | No evidence it is specific to reasoning |
| Any gain on REASON comes with a loss on TEXT | It trades general quality for the task |
| Swap heat map shows bands of low loss increase | Groups of layers are interchangeable (QE) |

Stop for review.

## 6. Budget

Unknown until task 0. Rough count: task 1 is 5 configurations per layer, task 2 about 60 to 70,
task 4 about 500 on a small set. At a guessed 25 seconds per full configuration this is 3 to 5
hours on the 14B model. If task 0 measures more than 60 seconds per configuration, halve the
SEARCH items for tasks 1, 2 and 4 and say so.

## 7. Limits

- One model family, one size. The original reports used larger models with more layers; a null
  result on a smaller one does not refute them.
- The model was never trained to be run this way. This measures slack in an existing model, not
  what a model trained with a loop could do (that is Experiment B).
- REASON is arithmetic only. A gain there is not a gain in "reasoning" in general.
- Interventions apply to every token position. Selecting positions is a possible follow-up.
