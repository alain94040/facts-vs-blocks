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

Every analysis also runs on two controls: the untrained initial weights, and a trained model whose
entries have been shuffled within each matrix. A finding counts only if it is clearly stronger in
the trained models than in both controls.

### Task 3: which parts grow with facts

Use the main runs. For each component (each attention head, each MLP neuron):

1. **Importance per data type.** Attribution = absolute value of the sum of weight times gradient
   over the component's weights, averaged over 2,000 examples. Compute once on story validation
   loss and once on the loss of answer tokens in biography prompts.
2. **Ablation curves.** In each MLP layer, replace neurons with their mean activation, in order of
   falling biography importance, at 1%, 2%, 5%, 10% and 20% of the layer. Record recall and story
   loss. The knowledge set is the smallest set of neurons whose ablation halves recall while
   raising story loss by less than 0.05 nats.
3. **Effective rank.** For each weight matrix, the exponential of the entropy of its normalised
   singular values.

Prediction if the split is real: the knowledge set sits in a few middle MLP layers and grows roughly
in proportion to the number of people; attention heads and story-critical neurons stay about the
same across main-0 to main-32k.

Output: `results/task3/summary.json` and three plots (biography importance by layer per N;
knowledge-set size against N; effective rank by layer per N).

### Task 4: swapping parts between models

Use fork-A and fork-B. Build hybrids that take most weights from A and some from B. Measure recall
on set A, recall on set B, and story loss.

| Hybrid | Weights taken from B | Expected if the split is real |
| --- | --- | --- |
| H1 | All attention layers | A's facts kept above 80% of original, story loss up less than 0.1 nats |
| H2 | All MLP layers | A's facts lost, set B facts appear |
| H3 | Only the MLP layers where Task 3 found the knowledge set | Facts swap from A to B, little story damage |
| H4 to H9 | One MLP layer at a time | Shows which layers hold facts |
| Control | All attention layers, from main-8k seed 1 instead | Should fail: unrelated coordinates |

Keep A's token embeddings in every hybrid. Run one extra hybrid that swaps only the embeddings and
report it separately.

Output: `results/task4/swaps.json` and one bar chart.

### Task 5: catalogue of attention heads

For every head in every model and both controls, on 500 story sequences:

- Previous-token score: average attention to the position just before.
- Induction score: on repeated random sequences, attention from a token to the token just after its
  earlier copy.
- Duplicate-token score: attention to earlier copies of the same token.
- Sink score: average attention to the first position.
- Copying score: share of positive eigenvalues of the head's OV circuit passed through the
  embeddings (Elhage et al. 2021, "A Mathematical Framework for Transformer Circuits").
- Spectral signature: top 16 singular values of W_Q W_K^T and of W_V W_O, each scaled by its largest.

Standardise all features, cluster heads with agglomerative clustering (Ward linkage), and report the
number of clusters K at distance thresholds 0.5, 1, 2 and 4. Name each cluster by its highest score.

For neurons: correlate each neuron's activations in seed 0 with every neuron of the same N in
seed 1 and record the best match. Report the fraction with best correlation above 0.8, by layer.

Prediction if the split is real: the same head types appear at every N with K roughly constant,
controls give no clean clusters, and neurons in knowledge layers are less universal across seeds.

Output: `results/task5/heads.csv`, `results/task5/clusters.json`, a plot of K against threshold per
model and control, and a plot of neuron universality by layer.

### Task 6: report

Write `results/REPORT.md` with the numbers and plots from Tasks 2 to 5. No interpretation beyond
marking which rows of this table apply:

| Result | What it means | Next step |
| --- | --- | --- |
| Knowledge set grows with N, attention and story parts stay fixed, H1 keeps facts, H3 moves facts | The split is real at this scale | Count block classes in the fixed part; repeat on real text |
| Knowledge spread across all layers, H1 or H3 damage both facts and stories | Facts and procedures share weights | Revise hypothesis: sparse model or separate memory layer |
| Head clusters recur at every N, controls show none | Reusable attention blocks exist | Measure what share of parameters they cover |
| Controls cluster as well as trained models | The method finds its own structure | Fix the method before concluding anything |
| Gate 2 failed | Models undertrained or broken | Fix training; no analysis is valid yet |

**End of Phase 2. Stop for human review.**
