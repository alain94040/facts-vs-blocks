# Facts vs building blocks: execution plan

A step-by-step plan to test whether a small language model splits into a fact store and reusable
building blocks. Each task has a deliverable and a numeric gate. Agent rules are in `CLAUDE.md`.

## Why this experiment

Working hypothesis: part of a language model is knowledge (facts, essentially random, not reusable)
and part is a set of unified building blocks (procedures that recur). We control the number of facts
a model must learn and observe which weights change with it.

Known weaknesses the design accounts for:

- Blocks that do the same job usually look different because of basis (symmetry) or real functional
  difference, not rounding noise. So compare behaviour on real activations, never raw weight values.
- Searching for structure finds structure in random weights too. So every analysis also runs on
  controls, and a finding only counts if it clearly beats them.
- Synthetic random facts are cleaner than real facts. Treat a clean split here as an upper bound.

## Phase 1: data and models

### Task 0: setup and smoke test

- Create folders `data/`, `model/`, `train/`, `analysis/`, `configs/`, `results/`.
- Write `model/gpt.py`: a standard decoder-only transformer with pre-LayerNorm, learned position
  embeddings, tied input and output embeddings, GELU MLP, no dropout. About 150 lines, nanoGPT style.
- Write `train/train.py`: AdamW (lr 1e-3, betas 0.9 and 0.95, weight decay 0.1), 1% linear warmup
  then cosine decay to 1e-4, float32 on MPS, gradient clipping at 1.0, batch of 256 sequences of
  256 tokens. Save checkpoints at 10%, 20%, 50% and 100% of steps.
- **Gate 0:** a 50-step run on random tokens finishes without errors, loss falls, and throughput
  (tokens per second) is recorded in `results/task0/throughput.json`.

### Task 1: data

**Stories.** Download `roneneldan/TinyStories` from Hugging Face. Train a byte-level BPE tokenizer
with vocabulary 4096 on the stories plus a sample of biographies; save to `data/tokenizer.json`.
Tokenize the train and validation splits separately.

**Synthetic biographies.** Write `data/make_bios.py`, generating people with six attributes chosen
uniformly and independently:

| Attribute | Choices | Example |
| --- | --- | --- |
| Birth date | Any day 1900 to 1999 | March 14, 1952 |
| Birth city | 200 invented city names | Varnholt |
| University | 300 invented names | Dellmore Institute |
| Major | 100 real subjects | geology |
| Employer | 250 invented company names | Quellbright |
| Employer city | Same 200 cities | Ostrava Bay |

- First and last names come from lists of 400 each; every full name must be unique.
- Write 5 sentence templates per attribute. Templates 1 to 4 are for training. Template 5 is held
  out for testing only.
- Each time a person appears in training data, pick a random training template for each attribute
  and shuffle the sentence order. (Needed for facts to be stored in an extractable form; see
  Allen-Zhu and Li, arXiv 2309.14316.)
- Each person appears 100 times over a run.
- Make two disjoint people sets with different names, set A and set B. Save every set as JSON with
  a fixed seed.
- **Gate 1:** write 5 random biographies and 5 random stories to `results/task1/samples.txt` and
  record the average token length of a biography. Stop: a human checks the samples before Task 2.

### Task 2: training runs

**Model:** 6 layers, width 256, 8 heads, MLP width 1024, context 256 (about 4.7M parameters outside
the embeddings). Each person carries about 53 bits, so 32k people is roughly 1.7M bits, well below
the model's estimated capacity of 5 to 10M bits.

### Task 2a: fix the training recipe (added after attempt 1 failed)

Attempt 1 (`results/task2_attempt1/`) failed Gate 2: main-2k reached 8% recall. The evaluation was
checked and is correct; the model did not memorise the facts. See `docs/LOG.md`. Two changes:

**Change 1: one token per entity.** In attempt 1 every invented word cost 4 to 5 tokens, so a name
was about 10 tokens and every value had to be spelled out. Fix:

- Add every pool word as its own token with `tokenizer.add_tokens`: the 400 first names, 400 last
  names, 200 city root words, 300 university root words and 250 employer root words. Real words
  (majors, months, "Bay", "Institute", "Systems" and so on) stay as they are. Keep the existing
  4096-token vocabulary and ids unchanged; the new tokens get ids from 4096 up. Set `vocab_size`
  in `configs/model_small.json` to the new total.
- Do not regenerate the pools or the people: `data/make_bios.py --seed 0` output stays identical.
- Requirement, checked by an assertion over every pool word in every template: each pool word
  encodes to exactly one dedicated id (a separate bare-space token in front of it is acceptable),
  and decoding round-trips to the original text.
- Re-run `data/prepare.py` and `data/samples.py`. Story tokenization must be unchanged apart from
  the tokenizer file; assert the story token counts equal those of Task 1.
- Write results to `results/task2a/`. Record the new mean biography length in tokens.

**Change 2: a declared pilot ladder.** Using main-2k seed 0 only, try these recipes in order and
stop at the first one that passes. No other recipe may be tried.

| Pilot | Exposures per person | Everything else |
| --- | --- | --- |
| P1 | 100 | as in Task 2 |
| P2 | 300 | as in Task 2 |

Pass = overall recall on training templates at least 0.80, and name-copy accuracy at least 0.90
(defined under Evaluations). The first passing recipe is frozen and used for every Task 2 run;
replace "100" exposures below by the frozen number (ctrl-8k-x4 uses four times it). If P2 also
fails, write `STOP.md` with both sets of numbers and stop.

Before each pilot, run 50 steps as usual. Each pilot is about 35 to 45 minutes.

### Task 2b: final recipe (added after the Task 2a pilots)

Pilot results are in `results/task2a/` and discussed in `docs/LOG.md`. P2 memorised the facts
(recall 0.955). Its name-copy miss is explained by memorisation competing with copying and does
not show a broken model. The held-out template test turned out to be badly designed. Changes:

**Change 1: exposures are 300** for every run (ctrl-8k-x4: 1,200).

**Change 2: template 5 is seen for half of the people.** In attempt 1 and Task 2a no model ever
saw template 5 for anyone, so it could not know which attribute an unfamiliar phrasing asks for.
New rule, for set A and set B alike: people with an **even** index in their set use all five
templates in training (a uniform random choice among 5 per attribute per exposure). People with
an **odd** index use templates 1 to 4 only, as before. Template 5 on odd-index people is the
held-out test: the wording is known, but never for that person.

**Change 3: one confirmation pilot, P3.** Build the main-2k stream with changes 1 and 2 and train
seed 0 (about 45 minutes). Results go to `results/task2/main-2k-s0/`; if it passes, this run *is*
main-2k seed 0. Pass = Gate 2 below applied to this run. If P3 passes, continue straight into the
remaining Task 2 runs without waiting for review. If it fails, write `STOP.md` and stop; no other
recipe may be tried.

**Data per run:** every run sees the same 300M story tokens in the same order. Every person
appears exactly 300 times (ctrl-8k-x4: 1,200). Total tokens differ per run; that is intended.
Biographies are spread uniformly over the whole stream at the document level, each preceded by
`<|endoftext|>`. Steps = stream length divided by tokens per step, with no wrap-around. Precision
is float32. Token counts below assume 77 tokens per biography; record the real ones.

| Run | People | Exposures | Seeds | About | Purpose |
| --- | --- | --- | --- | --- | --- |
| main-0 | none | - | 0, 1 | 300M tokens | Baseline: stories only |
| main-2k | first 2,000 of set A | 300 | 0, 1 | 346M | Small fact load |
| main-8k | first 8,000 of set A | 300 | 0, 1 | 485M | Medium fact load |
| main-32k | all 32,000 of set A | 300 | 0, 1 | 1,040M | Large fact load |
| ctrl-8k-x4 | first 8,000 of set A | 1,200 | 0 | 1,040M | Control: same biography volume and run length as main-32k, 4x fewer facts |
| fork-A | first 8,000 of set A | 300 | 0 | 485M | Swap experiment |
| fork-B | first 8,000 of set B | 300 | 0 | 485M | Swap experiment |

`ctrl-8k-x4` separates "more facts" from "more biography text and more training steps". In Task 3,
any difference claimed between main-8k and main-32k must also show between ctrl-8k-x4 and main-32k.

**Fork runs.** Both have the same total steps as main-8k and one LR schedule over those steps.
Trunk: the first 20% of steps read story tokens only (seed 0); save model and optimizer state.
Then continue twice from that checkpoint, at the same step and LR position: the remaining story
tokens in the same order, with set A (fork-A) or set B (fork-B) biographies mixed in uniformly.
Add `--init_from <checkpoint>` to `train/train.py`, restoring model, optimizer and step.

**Evaluations.** Add `train/evaluate.py`. During training, log story validation loss on a fixed
2,000-sequence sample every 5% of steps. At the end, save in each run's `metrics.json`:

- Story validation loss in nats per token, on the full validation split.
- Fact recall, exact match. Prompt = `<|endoftext|>` followed by the sentence prefix with its
  trailing space removed. Expected continuation = token ids of `" " + answer + "."`, obtained by
  tokenizing the full sentence and removing the prefix ids (do not tokenize the answer alone: the
  byte-level tokenizer attaches the space to the following word). Correct if greedy decoding
  reproduces those ids exactly. Report per attribute and overall, for three groups:
  `train_templates` (templates 1 to 4, all of the run's people), `template5_seen` (template 5,
  even-index people) and `template5_heldout` (template 5, odd-index people). Use all people for
  N up to 8k, a fixed random 8,000 for 32k.
- Fact loss: mean teacher-forced loss on the answer tokens, same prompts, same three groups.
- For runs with no biographies or a different people set, still report recall and fact loss on
  set A and set B. They should be at chance; this is the baseline for Task 4.
- Name-copy accuracy (in-distribution induction). Take people 16,000 to 16,999 of set B, who
  appear in no run. Render one training-template biography each (fixed seed). At every mention
  of the name after the first, check whether the model's top prediction after the first-name
  token is the last-name token. The model has never seen these people, so it can only succeed by
  copying from earlier in the context. Score at the bare-space position, as agreed in
  `QUESTIONS.md`. Report the accuracy, and split the errors into "last name of a person this run
  trained on" and "any other token" (the model's memory competing with the context).
- Random-token induction: 1,000 sequences of 50 random tokens repeated twice (random ids from the
  original 4096 excluding `<|endoftext|>`). Report loss on the second copy divided by loss on the
  first. Reported for every run, not a gate (see `docs/LOG.md` for why).

**Gate 2:** for every run, story validation loss changes by less than 0.02 nats between 90% and
100% of steps; every run with biographies reaches at least 80% overall recall on
`train_templates` for its own people (main-32k: at least 60%); every run with biographies has
name-copy accuracy of at least 0.50. This last number is a floor that catches a model with no
copying at all (chance is under 1%); it is not a quality bar. `template5_heldout` recall is
reported and not gated. If any condition fails, stop and report.

**Time:** about 6.3B tokens in total, roughly 12 to 13 hours at 136k tokens per second. Write the
recomputed estimate to `results/task2/plan_time.json` before starting. Run order after P3:
main-8k seed 0, main-32k seed 0, ctrl-8k-x4, main-0 seed 0, the fork trunk and both forks, then
the seed 1 runs. This order means that if the batch is interrupted, Tasks 3 and 4 can still be
done on seed 0. After each run, append its row to `results/task2/summary.csv` and commit.

**End of Phase 1. Stop for human review.**

## Phase 2: analysis

Rewritten on 2026-11-16 before any Phase 2 code existed; see `docs/LOG.md` for what was wrong
with the first version.

**Rules for this phase**

- Write and debug every analysis script on the throwaway pilot model `P2-main-2k-s0` from Task 2a.
  Only when a script runs end to end there, run it once on the real Task 2 models. Do not adjust
  a method after seeing real-model results; write to `QUESTIONS.md` instead.
- Each task names its own control. A finding counts only if it clearly beats that control.
- All evaluation data below is fixed with seed 0 and shared by every model.
- Linear algebra on CPU in float64.

**Datasets**

| Name | Content | What it needs from the model |
| --- | --- | --- |
| STORY | 2,000 fixed rows of the story validation split, loss on all tokens | Story language |
| FACT | Recall prompts (training templates) for the first 1,000 people of set A, loss on answer tokens. For fork-B use the first 1,000 people of set B | Stored facts |
| FORMAT | One biography each of set B people 16,000 to 16,999 (in no run). Loss on every token except attribute values and the first mention of the name | Biography wording and copying the name from context; no stored facts |

FORMAT exists because "important for biographies" mixes two things: the facts, and the procedure
of writing a biography. Only the first is knowledge.

**Components and ablation**

- MLP neuron: one of the 1,024 post-GELU activations in a layer (6,144 in the model).
- Attention head: its 32-dimensional output before the `out` projection (48 in the model).
- Ablating a component means replacing its activation at every position by its mean over a fixed
  reference sample (1,000 STORY rows plus the FORMAT biographies). Conclusions rest on real
  ablations with real forward passes.
- Importance of component c on dataset D, used only to rank components:
  `I_D(c) = | mean over tokens of (mean_c - a_c) * dL/da_c |`, a first-order estimate of the loss
  change from ablating c.

### Task 3: which parts hold the facts

Models: main-2k, main-8k, main-32k, ctrl-8k-x4, fork-A, fork-B (all seeds available). main-0 has
no facts and is used in Task 5 only.

1. **Rankings.** Compute `I_FACT`, `I_FORMAT`, `I_STORY` for all 6,144 neurons and 48 heads.
2. **Neuron ablation curves.** Ablate the top k neurons by `I_FACT`, over all layers together,
   for k = 8, 16, 32, ..., 4096. At each k record: recall on FACT, FORMAT loss, name-copy
   accuracy, STORY loss. Repeat with the top k by `I_STORY`, and with k random neurons (3 draws).
3. **Knowledge set.** K = the smallest k on the grid at which recall on FACT falls to half of its
   unablated value or below. The knowledge set is the top K neurons by `I_FACT`. Record K, its
   distribution over layers, and at that k the change in STORY loss and FORMAT loss, next to the
   same two numbers for K random neurons.
4. **Overlap.** Jaccard overlap between the knowledge set and the top K neurons by `I_STORY`, and
   by `I_FORMAT`. Report with the overlap expected for two random sets of size K.
5. **Head ablations.** Ablate each head alone: change in recall, FORMAT loss, name-copy accuracy,
   STORY loss.
6. **Effective rank** of each weight matrix: exp of the entropy of its normalised singular values.

Control: the random-neuron curves, and `ctrl-8k-x4` against main-32k.

Predictions if the split is real: K grows roughly in proportion to the number of people from 2k
to 32k; `ctrl-8k-x4` has K near main-8k, not near main-32k; ablating the knowledge set raises
STORY and FORMAT loss no more than ablating K random neurons; overlaps are near the random
expectation; no single head ablation removes most of the recall.

Output: `results/task3/<run>/metrics.json`, `results/task3/summary.csv`, and plots: ablation
curves per run; K against number of people; knowledge set by layer; effective rank by layer.

### Task 4: swapping parts between models

Models: fork-A and fork-B (shared trunk), and main-8k seed 1 as the unrelated model.

1. **Same-basin check.** Interpolate all weights between fork-A and fork-B at
   0, 0.25, 0.5, 0.75, 1 and record STORY loss at each point. Do the same between fork-A and
   main-8k seed 1. Report the barrier (highest loss on the path minus the mean of the two ends).
   If the fork-A to fork-B barrier is above 0.1 nats, the two forks have drifted apart: still run
   the hybrids, and flag that row in the report.
2. **Hybrids.** Base model A with parts from B, and the mirror image (base B, parts from A).
   "Attention" means the `qkv` and `out` weights and biases of each block. "MLP" means `up` and
   `down` weights and biases. Token embeddings, position embeddings and all LayerNorms stay with
   the base model. For every hybrid record recall on the base model's people, recall on the
   donor's people, FORMAT loss and STORY loss.

| Hybrid | Parts from the donor | Expected if the split is real |
| --- | --- | --- |
| H1 | All attention | Base facts kept above 80% of original, STORY loss up less than 0.1 nats |
| H2 | All MLP | Base facts lost, donor facts appear |
| H3 | Only the MLP layers that hold 80% of the base model's knowledge set (Task 3) | Facts swap, little story damage |
| H4 to H9 | One MLP layer at a time | Shows which layers hold facts |
| H10 | Token embeddings only | Reported separately |

Control: H1 and H2 with main-8k seed 1 as the donor. They should damage the model far more.

Output: `results/task4/interpolation.json`, `results/task4/swaps.json`, one bar chart.

### Task 5: do the same components recur

**Head features** (every head, every model including main-0, and the untrained initial weights).
main-0 and the untrained model never trained the 1,550 entity tokens, so any score computed on
FORMAT is meaningless for them. Compute the two STORY scores below for every model, and the FORMAT
scores only for models trained on biographies. Comparisons that include main-0 or the untrained
model use STORY-based features only:

- Story induction score: on STORY, at each position whose token occurred earlier in the same row,
  the attention mass on the positions right after those earlier occurrences.
- Story duplicate-token score: on STORY, at the same positions, attention mass on the earlier
  occurrences themselves.

- Previous-token score: mean attention to the position just before, on STORY.
- Sink score: mean attention to position 0, on STORY.
- Induction score: on FORMAT biographies, at each bare-space position after a repeated first
  name, the attention mass on earlier last-name tokens of that biography.
- Duplicate-token score: at each repeated first-name position, attention mass on its earlier
  copies.
- Copying score: with `W_OV` the head's 256 by 256 value-output map, take the eigenvalues of
  `W_OV W_E^T W_E` and report the sum of their real parts divided by the sum of their absolute
  values (Elhage et al. 2021).
- Spectral signature: the top 16 singular values of `W_Q W_K^T` and of `W_V W_O`, each divided by
  its largest.
- The four ablation effects from Task 3 step 5 (trained models only; zero for the untrained one).

No score uses random-token sequences: Phase 1 showed these are far outside what the models know.

**Head matching.** Standardise features over all heads of all trained models. For each head of
seed 0, find the nearest head (Euclidean) in seed 1 of the same run. Null distribution: the same
nearest-distance computed after shuffling each feature column independently across the seed 1
heads, 100 times. A head recurs if its distance is below the 5th percentile of the null. Report
the fraction that recur, for each run with two seeds, and between main-2k and main-32k of seed 0.
Also describe the pooled heads: k-means with 8 clusters, and per cluster its mean features and
how many models contribute at least one head.

**Neuron matching.** On a fixed 200,000 token positions (STORY and FORMAT mixed; STORY only for
main-0 and the untrained model), correlate every
neuron of seed 0 with every neuron in the same layer of seed 1 and keep the best match. Report
the fraction with best correlation above 0.5 and above 0.8, by layer, and separately for neurons
inside and outside the knowledge set. Null: best matches between seed 0 and the untrained model.

Predictions if the split is real: most heads recur across seeds and across fact loads; neurons
in the knowledge set recur less than the others; the null gives low recurrence.

Output: `results/task5/heads.csv`, `results/task5/head_matching.json`,
`results/task5/neuron_matching.json`, and plots of both recurrence measures.

### Task 6: report

Write `results/REPORT.md` with the numbers and plots from Tasks 2 to 5. No interpretation beyond
marking which rows of this table apply:

| Result | What it means | Next step |
| --- | --- | --- |
| K grows with people, ctrl-8k-x4 near main-8k, knowledge-set ablation spares stories and format, H1 keeps facts, H3 moves facts | The split is real at this scale | Count block classes in the fixed part; repeat on real text |
| Knowledge-set ablation damages stories or format as much as it damages facts, or H1 and H3 damage everything | Facts and procedures share weights | Revise hypothesis: sparse model or separate memory layer |
| K does not grow with people, or ctrl-8k-x4 looks like main-32k | "Knowledge set" tracks amount of text, not number of facts | The measure is not isolating facts; rethink it |
| Heads recur across seeds well above the null | Reusable attention blocks exist | Measure what share of parameters they cover |
| Recurrence is no better than the null | No evidence of shared blocks at this scale | Report as a negative result |
| Fork barrier above 0.1 nats | Swap results are not interpretable | Shorter fork, or align the models first |
| Gate 2 failed | Models undertrained or broken | Fix training; no analysis is valid yet |

**End of Phase 2. Stop for human review.**
