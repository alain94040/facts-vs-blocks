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
