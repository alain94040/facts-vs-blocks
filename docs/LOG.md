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
