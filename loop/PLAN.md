# Experiment B: more reasoning steps per weight by looping a block

A second, independent experiment. It asks whether one block of weights applied several times can
replace several different blocks for tasks that need many reasoning steps in a row.

The rules in `CLAUDE.md` apply. In addition:

- All code lives in `loop/`, all outputs in `results/loop/`, checkpoints in `checkpoints/loop/`.
  Do not edit any file outside `loop/` and `results/loop/`. In particular do not touch `model/`,
  `train/`, `data/` or the root `PLAN.md`.
- Do not start any run while another training run is using the GPU. Check with `ps` first.
- Commit messages: `loop task N: <what>`.

## 1. Question and measure

A transformer computes its answer in one pass through its layers, so the number of dependent
steps it can chain is limited by its depth. Looping a block adds depth without adding weights.

**Measure.** For a task where the number of steps k is a dial, `k*` is the largest k such that
accuracy is at least 90% at every step count from 1 to k. "Reasoning per weight" is `k*` divided
by the number of non-embedding parameters.

**Questions.**

- Q1. With the same depth and compute but half the weights, does a looped model reach the same
  `k*` as a model with all-different layers?
- Q2. Does running the loop more times at test time raise `k*`, including beyond the step counts
  seen in training?

Caveat known in advance: both tasks can in theory be solved in fewer layers than steps, by
combining steps in pairs (Sanford et al., arXiv 2402.09268). So `k*` is an empirical measure and
may exceed the depth. That does not affect the comparisons between models.

This experiment has no stored facts. It tests only the "looped reasoning block" half of the
design sketched in `docs/BACKGROUND.md`; combining it with a memory part depends on the outcome
of the main experiment.

## 2. Tasks

Both are synthetic, generated on the fly from a seed, with no memorisation possible: every
sequence is a fresh random problem. The model never sees its own outputs, so intermediate results
must be carried inside the network.

**T1, pointer chasing.** A random permutation of n = 32 nodes is written as 32 shuffled pairs,
then a start node, then k hop tokens. After the j-th hop token the model must output the node
reached after j hops.

```
<bos> a1 b1 a2 b2 ... a32 b32 <q> s > > > ... >
```

Targets: at the j-th `>` the target is the node j hops from `s`. No loss on other positions.

**T2, running arithmetic.** Arithmetic modulo 23. A start value, then k steps, each an operator
(`+`, `-`, `*`) and an operand (0 to 22 for `+` and `-`, 1 to 22 for `*`). After each operand the
model must output the running value.

```
<bos> x0 op1 a1 op2 a2 ... opk ak
```

Targets: at each operand position, the value after applying all steps so far.

**Step counts.** Training samples k uniformly from 1 to 24. Evaluation uses a fixed set (seed 0)
of 2,000 problems per step count for k = 1 to 32. Accuracy at step j is measured at the j-th
target position of problems with k = 32 for j up to 32 (all models are scored on the same
sequences). Steps 25 to 32 are beyond training.

Position embeddings must cover the longest sequence. Use learned absolute positions as in the
main experiment, with 8 extra hop or step positions initialised like the others (they are
untrained beyond k = 24; this is part of what Q2 tests and is reported, not fixed).

## 3. Models

Width 128, 4 heads, MLP width 512, pre-LayerNorm, GELU, no dropout, tied embeddings. One layer is
about 0.2M parameters. Write `loop/model.py`; a looped model applies one shared layer (including
its LayerNorms) r times in a row, with no signal telling it which iteration it is on.

| Name | Layers, in order | Depth | Layer weights | Purpose |
| --- | --- | --- | --- | --- |
| BASE-6 | 6 different | 6 | 6 | Reference |
| BASE-3 | 3 different | 3 | 3 | Same weights as the looped models, half the depth |
| BASE-12 | 12 different | 12 | 12 | Deep reference |
| LOOP-4 | 1 entry, 1 shared x4, 1 exit | 6 | 3 | Q1: depth of BASE-6, weights of BASE-3 |
| LOOP-R | 1 entry, 1 shared x r, 1 exit | r + 2 | 3 | Q2: r drawn uniformly from 2 to 8 for each training batch |

LOOP-R is evaluated at r = 2, 4, 6, 8, 10, 12 and 16. At r = 10 it has the depth of BASE-12 with
a quarter of the layer weights.

Training: AdamW, lr 1e-3 with 2% warmup and cosine decay to 1e-4, betas 0.9 and 0.95, weight
decay 0.1, gradient clipping 1.0, batch 256, float32 on MPS. Every model gets the same number of
steps and the same data stream for a given seed. Three seeds (0, 1, 2).

## 4. Tasks for the agent

### Loop task 0: code and smoke test

Write `loop/data.py` (both generators, with a brute-force checker of the targets),
`loop/model.py`, `loop/train.py`, `loop/evaluate.py`. Unit checks, all must pass:

- For 1,000 random problems per task, targets equal an independent step-by-step computation.
- LOOP-4 has exactly the non-embedding parameter count of BASE-3.
- LOOP-R at r = 4 with LOOP-4's weights gives identical logits to LOOP-4.
- 50 training steps of BASE-6 on each task: loss falls. Record steps per second.

### Loop task 1: pilot, to set the training length and check the dial

One seed (0), BASE-6 only, both tasks. Train for S = 20,000 steps, evaluating every 2,000 steps.

Checks, per task:

| Check | Condition | If it fails |
| --- | --- | --- |
| Learnable | accuracy at steps 1 to 4 is at least 90% | Train once more with S = 60,000. If it still fails, write `STOP.md` and stop |
| Not at ceiling | accuracy at step 24 is below 90% | The dial is too easy. Use the hard variant below and repeat the pilot once. If still at ceiling, write `STOP.md` and stop |
| Converged | `k*` is unchanged over the last three evaluations | Train once more with S = 60,000 and use that |

Hard variants, used only if "not at ceiling" fails: T1 with n = 64 nodes and k up to 48 (evaluate
to 64); T2 with modulus 97 and k up to 48 (evaluate to 64).

The S and task variant that pass are frozen for everything below. No other recipe may be tried.
Stop after the pilot and wait for review. This stop exists because the pilots of the main
experiment showed that first guesses at such settings are often wrong.

### Loop task 2: the runs

After review: 5 models x 2 tasks x 3 seeds = 30 runs with the frozen settings. BASE-6 seed 0 is
the pilot run and is not retrained. Order: all seed 0 runs first. Save a row per run to
`results/loop/summary.csv` after each run and commit.

Per run, save in `metrics.json`: accuracy at every step 1 to 32, `k*`, `k*` restricted to steps
1 to 24, non-embedding parameters, steps per second, and for LOOP-R the same for every test r.

### Loop task 3: report

Write `results/loop/REPORT.md` with: one table of `k*` (median and range over seeds) per model
and task; one plot per task of accuracy against step count for all models; one plot of `k*`
against test r for LOOP-R; and `k*` per million parameters. No interpretation beyond marking
which rows of this table apply:

| Result | What it means |
| --- | --- |
| LOOP-4 `k*` is within 10% of BASE-6 and clearly above BASE-3 | A looped block replaces different layers at half the weights (Q1 yes) |
| LOOP-4 `k*` is close to BASE-3 | The number of distinct weights limits reasoning here, not depth (Q1 no) |
| LOOP-R `k*` rises with test r up to the trained range | More loops buy more steps, within training (Q2 partly) |
| LOOP-R `k*` at r above 8 exceeds 24 | More loops extend reasoning beyond what was trained (Q2 yes) |
| LOOP-R at r = 10 matches BASE-12 | Same reasoning at a quarter of the layer weights |
| All models reach the same `k*` | The task did not separate them; the dial needs redesign |
| Seeds disagree by more than the gaps between models | No conclusion; more seeds needed |

"Clearly above" and "close to" mean: the ranges over the three seeds do not overlap, or do.

Stop for review.

## 5. Budget

Unknown until Loop task 0 measures speed. Design target: a BASE-6 run of 20,000 steps in about
10 minutes, so 30 runs in 5 to 8 hours (BASE-12 and LOOP-R at high r are slower). If the pilot
needs 60,000 steps, run seeds 0 and 1 only and say so in the report.

## 6. Limits

- Toy tasks. A positive result shows the mechanism is available, not that it helps a language
  model.
- Looping saves weights, not computation.
- Three seeds give a range, not a confidence interval.
- No iteration signal is given to the loop. A variant with one is a natural follow-up and is not
  part of this plan.
