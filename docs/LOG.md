# Lab log

Decisions and failed attempts, newest last. Written by the human reviewer side, not by the
executing agent.

## 2026-11-16: Task 2 attempt 1 failed Gate 2

Run: main-2k seed 0, 323.6M tokens, 37.8 min. Files in `results/task2_attempt1/`.

| Check | Threshold | Measured |
| --- | --- | --- |
| Story validation loss change, last 10% | < 0.02 nats | 0.012 (pass) |
| Fact recall, training templates | >= 0.80 | 0.083 (fail) |
| Random-token induction ratio | < 0.5 | 0.93 (fail) |

**Diagnosis.** The execution was correct: the agent confirmed that real greedy decoding matches
the teacher-forced evaluation, and outputs have the right form with the wrong value. The fault is
in the experiment design.

1. *Recall.* Fact loss on trained people was 0.69 nats per answer token against 1.13 for people
   never seen, so the model learned part of the facts but far from all. Likely causes: invented
   words cost 4 to 5 tokens each, so the model had to bind a 10-token name to values it must spell
   out token by token; and biographies were only 7% of the stream, 100 exposures each, while the
   model was still busy learning the stories. The reference setup this was modelled on
   (arXiv 2404.05405) used names of one or two tokens and trained on biographies alone.
2. *Induction.* Loss on the first copy of random tokens was 13.1, above the 8.3 of a uniform
   guess, so random token soup is far outside what the model has seen. The 0.5 threshold was a
   guess made without evidence that a TinyStories model of this size learns general-purpose
   induction. The check does not measure what it was meant to (that the model learned at least
   one known procedure in its own domain).

**Decisions.**

- Entities become single tokens (PLAN.md Task 2a, change 1).
- A two-step pilot ladder on exposures, declared in advance (change 2).
- The random-token induction ratio stays reported for every run but is no longer a gate. It is
  replaced as a gate by name-copy accuracy on unseen people, which tests copying from context
  inside the model's own domain. Threshold 0.90, set before any measurement of it exists.
- main-32k gets a lower recall threshold (0.60), set now, before it has ever been run.

**Honesty note.** Changing a gate after seeing a result is normally not allowed here. This is an
exception made openly by the reviewer for a prerequisite check ("is the model trained"), not for
a measurement that tests the hypothesis. Thresholds in Phase 2 are not to be touched this way.

## 2026-11-16: Task 2a pilots, and a second correction to the gates

Files in `results/task2a/`. Single-token entities worked as intended (biography 77 tokens, was 117).

| | P1 (100 exposures) | P2 (300 exposures) |
| --- | --- | --- |
| Recall, training templates | 0.643 | 0.955 |
| Name-copy accuracy on unseen people | 0.971 | 0.870 |
| Errors that are a trained person's last name | 71 of 146 | 591 of 651 |
| Recall, held-out template 5 | 0.067 | 0.186 |

**Reading.**

1. *Memorisation works at 300 exposures.* Birth date is the hardest attribute (0.84), as expected:
   it carries 15 of the 53 bits.
2. *The name-copy drop is memory competing with context.* The test people share first names with
   trained people (400 first names, 2,000 people, so about 5 trained last names per first name).
   A model that has memorised "First -> Last" for real people sometimes answers from memory
   although the context says otherwise. 91% of P2's errors are of exactly that kind, against 49%
   in P1, which memorised less. So the stronger the memory, the lower this score. The 0.90
   threshold I set was wrong because I did not foresee this: the check penalises the very thing
   the run is meant to achieve. It is kept as a floor (0.50) and its error split is now a
   reported measurement, because the competition between stored facts and in-context procedure
   is itself relevant to the research question.
3. *The held-out template test was ill-posed.* No model saw template 5 for anyone, so recall on
   it mostly measured whether an unseen phrasing could be mapped to the right attribute.
   Per attribute in P2: employer city 0.70 ("goes to work every day in", close to a training
   phrase), birth date 0.26, employer 0.15, the rest about 0. The fix is the standard design:
   show the wording for half of the people and test it on the other half.

**Decisions.** Exposures 300. Template 5 seen for even-index people. Name-copy gate becomes a
floor of 0.50. One confirmation pilot (P3), which doubles as main-2k seed 0.

**Honesty note.** This is the second time a Phase 1 check is changed after a result, both times
because the check I designed did not measure what I intended. Both changes are to checks of
"is the model fit to analyse", and both are recorded here with the numbers that prompted them.
No Phase 2 measurement or threshold has been run or changed. If P3 fails, the right move is to
stop and rethink the setup rather than adjust a third time.

## 2026-11-16: Phase 2 audited and rewritten before it ran

Two Phase 1 checks failed today because they did not measure what was intended. Phase 2 was
re-read with that in mind while the Task 2 batch trained. Nothing in Phase 2 had been run.

| Problem in the first version | Fix |
| --- | --- |
| "Biography importance" mixed stored facts with the procedure of writing a biography, so procedure neurons would have been counted as knowledge | Three datasets: STORY, FACT, FORMAT (unseen people, no value tokens) |
| Importance was "weight times gradient summed over a component's weights", which is ambiguous and unrelated to the mean-ablation used next to it | Activation-level estimate of the mean-ablation effect, used only for ranking; claims rest on real ablations |
| The knowledge set was defined per layer and required a story-loss bound that might have no solution | One global ranking, K at the point where recall halves, story and format damage reported next to a random-neuron baseline |
| No baseline for ablation damage | Random sets of the same size |
| Swaps assumed two forks stay in the same coordinates after 80% of training apart | Measured with a weight-interpolation barrier (Frankle et al., arXiv 1912.05671) |
| Swaps did not say what happens to LayerNorms and embeddings, and only went one way | Specified, and mirrored |
| Head scores used random-token sequences, which Phase 1 showed are out of distribution (loss 13 against 8.3 for a uniform guess) | Scores computed on biographies of unseen people |
| "Controls give no clean clusters" was ill-posed: random heads all look alike and form one tight cluster | Recurrence across seeds against a shuffled null |
| A blanket "shuffled weights" control was required for every analysis, although attribution on a model at chance loss means nothing | Each task names its own control |
| Methods could be tuned on the real models | All scripts are developed on the throwaway pilot model first |

**Known limits that remain.** Two seeds for the main runs and one for the forks, so no error
bars. Neurons are an imperfect unit because of superposition. Facts here are random by
construction. A clean result is an upper bound and needs a repeat on real text.

## 2026-11-16: Experiment B added (loop/PLAN.md)

John reframed the goal: not only compression, but more "thinking" at the same or a smaller number
of weights, by understanding what a block does well enough to reuse it deliberately rather than
repeating layers blindly. Experiment B is the smallest test of the reuse half: can one looped
block replace several different layers on tasks with a dial for the number of reasoning steps.
"Thinking strength" is defined there as k*, the longest chain of steps solved at 90% accuracy.

Design choices made because of today's failures: a pilot with a declared ladder sets the training
length and checks that the task is neither unlearnable nor at ceiling, and the agent stops for
review after the pilot. The plan states in advance that both tasks can in theory be solved in
fewer layers than steps, so k* is an empirical measure, not a count of layers.

## 2026-11-16: Experiment C added (repeat/PLAN.md)

John wants to study layer repetition on a real model without the bluntness of repeating whole
layers. Experiment C decomposes the intervention: recomputed second pass against a plain scaling
of the same update, attention-only against MLP-only, fractional doses, reasoning against recall
questions, and a layer interchangeability map. Guards: configurations are chosen on one half of
each question set and confirmed on the other, with paired bootstrap intervals; the difficulty of
the reasoning set is calibrated on the unmodified model only.

## 2026-11-17: Phase 1 complete, Gate 2 passed on every run

Numbers as reported by the executing agent (`results/task2/summary.csv`). Not yet interpreted
beyond what is needed to judge readiness for Phase 2.

| Run | Story loss | Recall, train templates | Template 5, held out | Name-copy |
| --- | --- | --- | --- | --- |
| main-0 s0 / s1 | 1.692 / 1.699 | - | - | 0.000 (not gated) |
| main-2k s0 / s1 | 1.692 / 1.693 | 0.960 / 0.951 | 0.940 / 0.927 | 0.897 / 0.853 |
| main-8k s0 / s1 | 1.696 / 1.696 | 0.973 / 0.970 | 0.961 / 0.959 | 1.000 / 1.000 |
| main-32k s0 / s1 | 1.747 / 1.747 | 0.688 / 0.773 | 0.685 / 0.763 | 0.999 / 1.000 |
| ctrl-8k-x4 s0 | 1.693 | 0.997 | 0.994 | 1.000 |
| fork-A s0 | 1.701 | 0.987 | 0.982 | 1.000 |
| fork-B s0 | 1.700 | 0.983 | 0.979 | 1.000 |

**Notes for Phase 2.**

- Facts are stored in a usable form: held-out wording is within 2 points of trained wording.
- main-32k is not saturated (0.69 and 0.77, a large seed gap) and its story loss is 0.05 nats
  worse than every other run. ctrl-8k-x4, with the same volume of biography text, shows no such
  cost, so the cost tracks the number of facts. This is the first sign that facts and stories
  compete for capacity at this model size; Phase 2 should check where.
- main-2k name-copy is lower than every larger run. With 2,000 people a first name predicts a
  few trained last names; with 8,000 or more it predicts too many to be useful, and the model
  copies from context. Consistent with memory competing with context when memory is cheap.
- main-0 scores 0 on name-copy because the entity tokens are untrained in that model. Phase 2
  Task 5 used FORMAT data for main-0, which would have been meaningless. Fixed in `PLAN.md`
  before Phase 2 started.

## 2026-11-17: Phase 2 reviewed; Phase 2b added

Results in `results/REPORT.md` (commits e57c2e6 to 422ee15). The agent marked the decision table
literally and left two points to the reviewer. Reviewer's calls:

- **"Facts and procedures share weights": yes.** The forks sit in one basin (barrier 0.006), yet
  no partial swap moves any donor fact: donor recall stays at or below 0.017 even when every MLP
  is swapped, and swapping only the token embeddings removes base recall (0.006). Facts are a
  joint code over embeddings, attention and MLPs, not a module that can be lifted out.
- **The Task 3 knowledge set is an entry stage, not storage.** Each first-name token is shared by
  5 (2k) to 80 (32k) people, so the model must bind first and last name into one person before
  any lookup. The knowledge set sits in layer 0, shrinks as people grow, recurs across seeds more
  than other neurons (0.63 above 0.5 at 32k, against 0.07 outside), damages FORMAT and name-copy,
  and one layer-0 head can remove 93% of recall. All of that fits a shared binding stage. The
  random baseline was not layer-matched, which exaggerated the contrast. Per-person content is
  more likely in MLP layers 3 and 4: swapping either alone drops base recall to 0.25 to 0.29.
  Nanda, Rajamanoharan et al. (2023, "Fact Finding") describe the same staging in real models.
- **The head null was too weak** (raised by the agent). Shuffling each feature column breaks the
  correlations among the 32 spectral values, so any real head beats it, and every pair recurred
  at 0.98 to 1.00. The STORY clustering is still informative: the untrained heads formed their
  own cluster, and each of the other 7 clusters has heads from all 11 trained models.
- **Task 5 choices made during the run** (positions split, story-only mean reference for main-0):
  accepted.
- **Neuron universality** is 2.5 to 8% above 0.5 correlation, in line with the 1 to 5% reported
  for GPT-2 (arXiv 2401.12181).

**What it means for the hypothesis.** A clean "random knowledge part plus block library" split
does not appear even with perfectly random facts, the easiest case for it. What appears instead
is a pipeline: a shared stage that binds a name to a person, middle MLPs that carry per-person
content, and a read-out stage. The stages may be the reusable blocks, with the facts written
across them.

**Phase 2b** tests that reading directly: where per-person content sits (layer-matched baseline,
per-person attribution and ablation), whether each attribute has a shared linear read-out
(probes on unseen people, and transfer between forks), where new facts can be written (fine-tune
one part of fork-A on new people), and a fair head null with a data-driven count of head types.

**Repo note.** During Phase 2 a local merge of `loop-plan` into `main` stopped on a conflict in
this file, and the Task 4 commit staged files into it by mistake. The agent restored them from
git's stored copies without rerunning anything. The merge was not completed; `loop-plan` (PR #5)
was later updated to merge cleanly on GitHub.

## 2026-11-17: Experiment C gets a weight comparison (Repeat task 4b)

John asked whether layer repetition is the best evidence of something abstract in the middle
layers, and proposed looking for what those layers have in common. A gain from repeating a layer
mainly shows that the layers share one residual format, so one layer's output is valid input for
another; it is weak evidence about reasoning. Task 4 already tests interchangeability by
behaviour. Task 4b adds the direct test: align the neurons and heads of two layers (their order
is arbitrary, the residual coordinates are shared), compare the aligned weights against a
shuffled null, guard against a few outlier residual coordinates, check with a planted positive
control that the method can find matches at all, and ask whether the layers that swap well are
the ones whose weights match, beyond the effect of distance. Weights only; about an hour on the
Mac.

## 2026-11-17: Phase 2b reviewed; Phase 2c fixes the probes

Results in `results/REPORT_2b.md` (commits 195a037 to a9122f9). Reviewer's reading:

- **Per-person content is a distributed code.** Important neurons each serve 25 to 45% of the
  people (median participation ratio 50 to 90 of 200); person-specific neurons barely exist (0 to
  2 per model). Removing a person's own top 32 neurons in layers 1 to 5 costs that person 10 to
  30% of recall and others about 3%. More repetition sharpens the code: ctrl-8k-x4 (4x exposures)
  has median PR 21 to 39 and an own-to-other ratio of 22.
- **Facts can be written almost anywhere after layer 0** (Task 4b). Training only MLP layers 3
  and 4 (1.05M weights) stores 1,000 new people at 0.987 with no story cost; attention alone
  reaches 0.93, embeddings alone 0.79, layer 0 alone 0.02. Where facts live is decided by training,
  not fixed by the architecture. Every variant forgot the old people (0.987 to 0.02 to 0.16).
  That was guaranteed by the design, which had no replay of old people, so the retention
  criterion was unfair; it is a planning error, not a finding.
- **The probe result is not trusted.** The last residual is linearly decodable by construction
  (the model's own unembedding does it at 0.96), yet the probes scored about 0.01. With about 4
  examples per class and no positive control, the probe design is the likely cause. Phase 2c
  remeasures with the model's own read-out.
- **Heads.** With the Gaussian null about half the heads recur across seeds (0.46 to 0.67), 0.25
  across fact loads, none against the untrained model; matched heads share a layer 2.5 to 3
  times more often than chance. Silhouettes beat the null by only 0.03 to 0.09: a continuum with
  weak structure, not a small set of discrete types.

**What it means.** Dissecting a trained model does not reveal a clean facts-versus-procedures
split, but Task 4b shows the split can be imposed: facts go where training puts them. The next
step in that direction is a model built with a designated memory, studied with its facts held
apart.

## 2026-11-17: Experiment C switches its main model to 7B

John pointed out that the 7B model runs about twice as fast. Tasks 1 to 4 evaluate hundreds of
configurations, so speed matters more than depth for the search. The 7B model (28 layers) still
gives enough middle layers to map. 14B (48 layers) moves to a confirmation step (Repeat task 5b):
the configurations selected on 7B are moved by fraction of depth and evaluated once on the REPORT
halves, with no search on 14B. Also recorded: `transformers`, `safetensors` and
`huggingface_hub` are allowed for Experiment C only, to load Qwen; the forward pass stays in
`repeat/flow.py`, and no quantisation or `accelerate`.

## 2026-11-17: Experiment C task 0 reviewed; RECALL-HARD added

Task 0 on Qwen2.5-7B passed: exact against the library, 3,669 tokens per second (13.8 s per
configuration), REASON at k = 3 gives 38% accuracy, TEXT loss 2.604. RECALL is at 100% accuracy,
so a gain could not show in accuracy. Changed before task 1, with REPORT untouched: a fourth set,
RECALL-HARD (atomic numbers in both directions, 236 items), is added and QD is judged on it.
Accepted as reported: RECALL has 261 items (disputed or ambiguous capitals removed), and the gain
and partial combinations are computed in float32 for exactness. `transformers` 5.23.0 and
`safetensors` 0.8.0 are pinned in `repeat/requirements.txt`.

## 2026-11-17: Experiment C tasks 1 to 4b reviewed

- **Blocks.** Repeating layers 8 to 11 raises REASON log-probability by 0.5 with no cost to TEXT
  or recall; doubling their update (`gain(2)`) hurts instead. The dose curve is smooth, and a
  quarter of a second pass (`partial(0.25)`) is the only SEARCH setting that improves REASON
  accuracy, TEXT and RECALL-HARD together. Added to task 5 before REPORT was opened.
- **Swaps.** Layers 7 to 11 can run in place of layers 12 to 17 almost for free (+0.005 to +0.018
  nats), the same region where repetition helps. Layers 18 and later are not interchangeable.
- **Weights.** Within each distance, pairs with more neuron matches swap more cheaply (mean
  Spearman -0.72, p < 0.001). Absolute match fractions are not trusted: the shuffled-coordinate
  null was too weak (flagged by the agent; the same flaw as Phase 2 Task 5, repeated by the
  reviewer). Task 4c adds a covariance-matched null and a mean-removed comparison.

## 2026-11-17: Experiment C reviewed; Experiment D added

Report in `results/repeat/REPORT.md`. Reviewer's calls on the two open points:

- **QC: accuracy decides.** Log-probability is a secondary score. On REPORT, re-running layer
  14's MLP lowers REASON accuracy (−0.09) and re-running its attention does not (+0.03, not
  significant). So repeating the MLP harms answers; neither half improves them.
- **"Neighbours share units": read as no.** The match measures higher-than-chance similarity,
  and mutual best matches are rarer than under the covariance null. Layers share directions,
  not units. The head result is uninformative: matched cosines of 0.05 to 0.07 cannot support
  "match fractions near 1". What stands is the link: weight similarity predicts swap cost
  (within-distance Spearman about −0.70).

**Overall.** On 7B, repeating layer 14 or block 8–11 raises the answer's log-probability but
never accuracy. None of the REASON gains carry over to 14B at the mapped depth; they reverse.
What carries over is smaller and structural: recomputing beats scaling the update, and a quarter
pass through the family lowers TEXT loss slightly on both models. Repetition tricks are
model-specific; the zone structure (entry, a swappable family, a specialised output stretch) is
the more durable finding, and it is not yet tested on 14B.

**Experiment D** (`basis/PLAN.md`) measures the family's shared directions directly: one
residual-side basis per matrix type, its kept energy against each layer's own basis and against
a rotated null, the effect of projecting all family layers onto it, and the compression that
implies, on 7B and 14B. Weights only, about 1.5 hours.
