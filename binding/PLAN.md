# Experiment F: where a person is assembled, and whether that stage is a table

An independent experiment on the Phase 1 toy models (6 layers, width 256, pre-LayerNorm, GELU
MLP). It runs forward passes and trains small sparse "transcoders" on activations. It does not
train or fine-tune the models themselves.

**Why.** Experiments C, D and E looked for a repeated operation at the level of whole layers of
Qwen and found none. This experiment moves down to features. It also returns to the toy model,
where every person and every fact is known. It asks the core question of the project as a count:

1. **Where do two name tokens become one person?** A person is a first-name token, a bare-space
   token and a last-name token. Step 1 of the usual picture is generic: attention copies the
   first name to the last-name position. Step 2 is the stage that responds to *this pair* and
   not to either name alone. The **interaction** of four names measures where step 2 happens:

   `I = r(F1 L1) − r(F1 L2) − r(F2 L1) + r(F2 L2)`

   where `r` is an activation at the last-name position. If the model only added the two names,
   `I` would be 0.
2. **Is that stage a table?** Replace each MLP by a wide sparse transcoder, so that most
   superposition is removed, and count the units that fire for one or a few people only. If
   facts are a table and procedures are shared blocks, then:
   - the number of person units grows with the number of people (2k, 8k, 32k);
   - `ctrl-8k-x4` sits with main-8k;
   - the shared units stay about the same, and recur across models trained on different fact loads.
3. **What do the shared units look like?** For each unit, measure its fan-in, and whether it acts
   like an AND (needs all its main inputs) or an OR (any one is enough).

The rules in `CLAUDE.md` and the Phase 2 rules apply:

- Develop every script on the pilot `P2-main-2k-s0`, then run it once on the real models. Its
  numbers are not results.
- Do not change a method after seeing real-model results; write to `QUESTIONS.md` instead.

In addition:

- All code lives in `binding/`, all outputs in `results/binding/`. Import `model/gpt.py`,
  `analysis/common.py`, `data/make_bios.py` and `train/evaluate.py`; do not edit them.
- Plain PyTorch and numpy only. Linear algebra on CPU in float64. Models in float32 on MPS.
- Commit messages: `binding task N: <what>`.
- Seed 0 for every random choice unless stated.

## Definitions

**Models.**
- Real: main-2k, main-8k, main-32k (seeds 0 and 1), ctrl-8k-x4, fork-A.
- Control: main-0 seed 0 (no people). Its entity tokens were never trained, so it is used only
  where stated.

**People.**
- The run's **trained people** are its own set (set A; set A for fork-A).
- **Unseen people** are set B people 16,000 to 16,999. They are in no run, are rendered with the
  same name pools, and are the FORMAT people of Phase 2.

**Name position.** A name is three tokens: first name, bare space, last name. "The last-name
position" is the position of the last-name token.

**Contexts.** Two prompt prefixes, both ending at the last-name token:
- START: `<|endoftext|>F L`;
- MID: `<|endoftext|>The employer of F L`.

Every measurement below is made in both contexts and reported for each.

**Measurement points.** The residual stream at the last-name position:
- after the embeddings (point `e`);
- after the attention of layer l (`aL`);
- after the MLP of layer l (`mL`).

That is 13 points: e, a0, m0, …, a5, m5. Also the 1,024 post-GELU activations of each MLP at that
position.

**Quadruples.**
- A **real quadruple** is (F1, L1, F2, L2) where F1 L1 is a trained person of the run, and none of
  F1 L2, F2 L1, F2 L2 is a person in sets A or B of any run.
- A **control quadruple** is the same with F1 L1 also not a person in any set.

Draw 500 of each per run: F1 L1 from the run's people 0 to 999, and F2, L2 uniformly from the name
pools until the conditions hold. Control quadruples use F1 and L1 drawn uniformly from the pools.
The draws depend only on the seed and the run's people list.

**Interaction ratio.** For a quadruple and a point, `IR = ‖I‖ / D`, where D is the mean of `‖r(x) − r(y)‖`
over the 6 pairs of the quadruple's four names. IR is 0 for purely additive names and of order 1
when the pair behaves like a new item.

## Task 1: where a person is assembled

Models: main-2k, main-8k, main-32k (seeds 0 and 1), ctrl-8k-x4, fork-A.

1. **Interaction by point.** For every point and context, report the median IR over the 500 real
   quadruples and over the 500 control quadruples. Also report the median of the difference
   (real − control) with a bootstrap 95% CI over quadruples, 2,000 resamples.
   - `P_bind` = the first point in order (e, a0, m0, …) where that difference is at least 0.1 and
     its CI excludes 0 (or `none`).

   Control quadruples measure the nonlinearity that any pair of names gets. The real − control
   difference is the part that exists because the pair is a known person.
2. **The routing head.** In the START context, for every head, report the mean attention from the
   last-name position to the first-name position, over the 1,000 trained people's names.
   - **Routing heads** are those with mean attention at least 0.3.
   - Mean-ablate all routing heads together (Phase 2 ablation, Phase 2 reference means) and redo
     step 1 for the real quadruples.
   - Report the real − control difference at `P_bind` and at m5, ablated vs not.
3. **Ablating the layer where binding happens.** If `P_bind` is an MLP point mL, mean-ablate all
   neurons of that MLP and report the same two numbers, and FACT recall on people 0 to 999.
4. **Neurons.** For every MLP layer, every neuron n and every quadruple, compute the neuron
   interaction `i_n` (the formula for `I` applied to n's post-GELU activation).
   - A neuron is **binding for a real quadruple** if `|i_n|` is above the 99th percentile of `|i_n|`
     over the 500 control quadruples, for that neuron and context.
   - Report, per layer and context:
     - the mean number of binding neurons per real quadruple;
     - the same count for control quadruples, which is about 10 by construction and is the null;
     - the number of distinct neurons that are binding for at least one real quadruple;
     - the histogram of, per neuron, the number of the 500 people it is binding for.

Output: `results/binding/task1/<run>/metrics.json`, `results/binding/task1/summary.csv`, and one plot
per context: median IR against point, real and control lines per run.

## Task 2: transcoders

A transcoder for MLP layer l is a wide sparse network trained to imitate that MLP:
- input: `h = ln(x)`, the MLP's own LayerNorm output;
- target: the MLP's output `y`, before it is added to the stream.

Architecture (Gao et al. 2024, "Scaling and evaluating sparse autoencoders", top-k):
- `pre = W_enc h + b_enc` (16,384 units);
- `a = ReLU(top-k(pre, k = 32))`, keeping the 32 largest values and setting the rest to 0;
- `ŷ = W_dec a + b_dec`.

Training:
- Decoder columns have unit norm (renormalise after every step). `W_enc` is initialised to
  `W_dec` transposed. `b_dec` starts at the mean of `y`.
- Loss: mean squared error divided by the variance of `y`.
- Adam, learning rate 5e-4, batch 4,096 positions, 20M positions. Linear warmup over the first
  1% of steps, then constant.
- The six layers' transcoders train together from one stream of forward passes, one per layer,
  each on its own MLP.

**Stream.** Sequences of 256 tokens drawn fresh with a seeded generator:
- 50% STORY: train split, never the Phase 2 STORY rows;
- 50% biographies of the run's trained people, rendered with training templates as in Task 2.

main-0 uses STORY only. Activations are never stored.

**Fidelity (gate).** Measured on held-out data (1,000 STORY rows not in the stream, and FACT
prompts for people 0 to 999):
- FVU, the fraction of `y`'s variance unexplained, per layer;
- dead units: those that never fire on 1M held-out positions;
- with one layer's MLP replaced by its transcoder at a time: FACT recall and STORY loss;
- the same with all six replaced together.

Gate F1, for each layer replaced alone:
- FVU at most 0.15;
- FACT recall at least 0.85 times the unedited model's;
- STORY loss up by at most 0.10 nats;
- dead units at most 30%.

On the pilot model only, a declared ladder:
- T1 is the recipe above.
- If T1 fails F1, try T2: 32,768 units, k = 64, 40M positions.
- If T2 fails too, write `STOP.md` with both sets of numbers and stop.

The first passing recipe is frozen for every model. On real models, F1 is reported, and a layer
that fails is flagged in every table that uses it. It is not retrained.

**Models.**
- One transcoder seed (0) for: main-2k, main-8k, main-32k (seeds 0 and 1), ctrl-8k-x4, fork-A,
  and main-0 seed 0.
- A second transcoder seed (1) for main-8k seed 0 only. It measures how stable transcoders are.

Output:
- transcoders in `checkpoints/binding/<run>/tc<seed>/L<l>.pt` (not committed);
- fidelity in `results/binding/task2/<run>/metrics.json` and `results/binding/task2/fidelity.csv`.

## Task 3: person units and shared units

For every transcoder unit u of every run, collect:

- **People it fires for.** At the last-name position in the START context, for each trained
  person:
  - all people for N up to 8k;
  - a fixed random 8,000 for main-32k;
  - also, separately, for each of the 1,000 unseen people.

  `P(u)` is the set of trained people for which u is active (a > 0); `P_unseen(u)` is the same
  for unseen people.
- **STORY rate.** `S(u)`, the fraction of positions where u is active, over 1M held-out STORY positions.
- **Name explanation.** Over `P(u)`, the share of people with the most common first name, and
  the share with the most common last name. Call the larger one `name_share(u)`.

**Classes**, assigned in this order:

| Class | Rule | Meaning |
| --- | --- | --- |
| DEAD | never active on held-out data | — |
| STORY | `S(u) ≥ 1e-3` | used by ordinary text |
| NAME | `\|P(u)\| ≥ 3` and `name_share(u) ≥ 0.8` | responds to one first or last name, whatever the pair |
| PERSON | `1 ≤ \|P(u)\| ≤ 10`, not NAME | responds to a few specific people |
| BIO | everything else that is active on trained people | biography-wide features |
| OTHER | the rest | — |

**Null for PERSON.**
- Take the same number of trained people as unseen people (a fixed 1,000, seed 0).
- Count the units that would be PERSON when `P` is computed on those 1,000 trained people.
- Count the same with `P_unseen` on the 1,000 unseen people.

A unit can look person-specific by chance (sparsity alone makes some units fire rarely), and the
unseen people show how often that happens. The ratio of the two counts is the measure.

**Report**, per run and layer:
- the count of each class;
- the PERSON count on 1,000 trained people next to the unseen null;
- the fraction of trained people (of all sampled) with at least one PERSON unit active at their
  last-name position.

**Recurrence of shared units.** For STORY units, record each unit's activation on 200,000 fixed
held-out STORY positions (the same positions for every run). For each pair below, and each STORY
unit of the first transcoder, find the best-correlated STORY unit of the second in the same
layer. Report the fraction of units with a best correlation of at least 0.8.

| Pair | Role |
| --- | --- |
| main-8k s0 tc0 vs main-8k s0 tc1 | Ceiling: same model, different transcoder seed |
| main-8k s0 vs main-8k s1 | Same data, different model seed |
| main-2k s0 vs main-32k s0 | Different fact load |
| main-0 s0 vs main-32k s0 | No facts vs many facts |
| main-8k s0 vs untrained-model transcoder | Null; train one tc0 on the untrained initial weights of seed 0 with the same recipe, STORY stream only |

Output: `results/binding/task3/<run>/units.csv` (one row per unit: layer, class, |P|,
|P_unseen|, S, name_share), `results/binding/task3/summary.csv`,
`results/binding/task3/recurrence.json`. Plots:
- PERSON count against number of people (2k, 8k, 32k, with ctrl-8k-x4 marked), one line per
  layer, next to the unseen null;
- class counts by layer for each run.

## Task 4: gate signatures

For main-8k seed 0, for every unit u of every layer's transcoder that is active on at least 100
held-out positions (STORY and biographies mixed, 1M positions).

**Inputs of u at a position.** The residual stream entering MLP l is a sum of vectors:
- the token embedding;
- the position embedding;
- each head output of layers 0 to l (after its slice of `out`, so 8 per layer);
- each active transcoder unit's decoder contribution for layers below l, `a_v W_dec[:, v]`.

These are u's inputs.

**Contributions.** The MLP's LayerNorm is applied with its per-position scale held fixed (divide
by that position's standard deviation; the mean subtraction is linear). Each input's contribution
to u's pre-activation is then exact and linear: `c_v = W_enc[u] · LN_fixed(v)`. Also report the
share of the residual the inputs leave out: the error of the replaced MLPs, measured as the norm
of the true residual minus the sum of inputs, divided by the true residual's norm.

**Per unit**, over the positions where u is active:
- **Fan-in:** the smallest number of inputs whose positive contributions add up to 80% of the
  total positive contribution, averaged over positions.
- **AND index:** among the 5 largest positive contributors at a position, the fraction whose
  removal alone (subtract `c_v`) drops u below its top-k threshold at that position. The threshold
  is the 33rd-largest pre-activation there. Average over positions.
  - 1 means every main input is needed (AND).
  - 0 means no single input is needed (OR, or redundant inputs).
- **Inhibition share:** the total negative contribution divided by the total absolute
  contribution.

**Report**, per layer and per Task 3 class: the median and IQR of each signature.

**Null:** the same signatures for the transcoder of the untrained model (Task 3 null). If layers
of the untrained model have no units active on 100 positions, report that.

Output: `results/binding/task4/signatures.csv` and one plot per signature: a box per class per layer.

## Task 5: report

Write `results/binding/REPORT.md` with the tables and plots, and mark which rows apply:

| Result | What it means |
| --- | --- |
| Gate F1 fails on the pilot after the ladder | Stop; Tasks 3 and 4 are void |
| `P_bind` is m0 in every run, and ablating routing heads cuts the real − control difference at `P_bind` by at least 75% | A person is assembled by layer 0: one routing head step, then the layer-0 MLP |
| `P_bind` is later than m0 in some runs | Assembly continues past layer 0 (report which point) |
| Binding neurons per real quadruple at least 3 times the control count in the binding layer, and no neuron is binding for more than 50% of people | The pair is detected by a spread of conjunction neurons, not one unit per person |
| PERSON count on trained people at least 5 times the unseen null in some layer, for every real run | Removing superposition reveals person-specific units |
| That PERSON count rises at least 2× from main-2k to main-8k and from main-8k to main-32k, and ctrl-8k-x4 is within 25% of main-8k | The person-specific part grows with the number of facts: a table |
| PERSON count on trained people below 2 times the unseen null in every layer | Even without superposition, people are a shared code, not a table |
| STORY-unit recurrence main-2k vs main-32k is at least 80% of the transcoder-seed ceiling, and the untrained null is below 30% of it | Shared units are the same whatever the fact load: shared building blocks |
| Median AND index of PERSON units at least 0.2 above that of STORY units | Person units are conjunctions; shared units are more OR-like |

Stop for review.

## Budget

- Task 1: forward passes on a few thousand short prompts per run, minutes in all.
- Task 2: 8 models plus the untrained null plus one extra seed, about 10 transcoder runs of 20M
  positions. Each is a forward pass of the toy model plus six small networks, about 3 to 6
  minutes. About 1 hour.
- Tasks 3 and 4: minutes per run.
- Total about 1.5 hours.

## Limits

- A transcoder is an approximation. Its error (FVU) is where it misses the MLP, and anything it
  misses is invisible to Tasks 3 and 4. Gate F1 bounds that.
- Units are not unique. A wider dictionary splits them further, so counts are only comparable
  between runs at the same width and k, which this plan fixes.
- At 16,384 units, main-32k cannot have one unit per person. A table would show as person units
  filling a growing share of the dictionary, and as a growing fraction of people covered, not as
  32,000 units.
- Task 4 ignores the transcoder error and how attention's inputs depend on other positions. It
  describes how a unit combines what reaches it, not where that came from.
- The toy model's facts are random and its people are invented. A clean table here is an upper
  bound for real models.
