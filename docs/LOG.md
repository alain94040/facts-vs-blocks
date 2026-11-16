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
