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

**Data per run:** the same 300M story tokens in the same order for every run, plus the biographies,
shuffled in at the document level.

| Run | People | Seeds | Purpose |
| --- | --- | --- | --- |
| main-0 | none | 0, 1 | Baseline: stories only |
| main-2k | 2,000 from set A | 0, 1 | Small fact load |
| main-8k | 8,000 from set A | 0, 1 | Medium fact load |
| main-32k | 32,000 from set A | 0, 1 | Large fact load |
| fork-A | 8,000 from set A | 0 | Swap experiment |
| fork-B | 8,000 from set B | 0 | Swap experiment |

The fork runs share a trunk. Train on stories only for 20% of the steps from seed 0 and save that
checkpoint. Then continue it twice, once adding set A and once adding set B, with the same story
order. A shared starting point keeps the two models in the same internal coordinates, so swapping
parts is meaningful without extra alignment.

**Evaluations, saved in each run's `metrics.json`:**

- Story validation loss in nats per token.
- Fact recall: for each person and attribute, prompt with the sentence up to the attribute and check
  whether greedy decoding produces it exactly. Report separately for training templates and for
  held-out template 5.
- Induction check: 1,000 sequences of 50 random tokens repeated twice. Report loss on the second
  copy divided by loss on the first.

**Gate 2:** every run's loss curve is flat over its last 10% of steps; main-2k reaches at least 80%
recall on training templates; every run's induction ratio is below 0.5. If any fails, stop and
report.

**Time:** each run is 300 to 600M tokens. Guess: 100k to 200k tokens per second, so 30 to 90 minutes
per run and 6 to 12 hours for all ten. If Task 0 measures under 75k tokens per second, cut story
tokens from 300M to 150M for every run and note it in the report.

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
