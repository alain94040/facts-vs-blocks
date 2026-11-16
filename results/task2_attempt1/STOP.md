# STOP: Gate 2 failed (Task 2)

Per PLAN.md Task 2 run order, main-2k seed 0 was trained first, alone. It fails Gate 2, so no
other Task 2 run was started (`train/run_task2.sh` is written but has not been run).

Run: `main-2k-s0`, 4,918 steps, 323.6M tokens, float32, 143.1k tokens/s, 37.8 min.
Command: `results/task2/main-2k-s0/cmd.txt`. Full metrics: `results/task2/main-2k-s0/metrics.json`.
Gate check: `results/task2/gate2/metrics.json`.

| Gate 2 condition | Threshold | Measured | Result |
| --- | --- | --- | --- |
| Story val loss change, 90% -> 100% of steps | < 0.02 nats | 0.0118 (1.7141 -> 1.7023) | pass |
| main-2k overall recall, training templates (set A, 2,000 people) | >= 0.80 | 0.0827 | **FAIL** |
| Induction ratio (loss 2nd copy / 1st copy) | < 0.5 | 0.932 (12.17 / 13.06) | **FAIL** |

Other numbers for this run:

- Story validation loss, full split: 1.6993 nats/token.
- Recall, training templates, per attribute: birth_date 0.010, birth_city 0.081, university 0.089,
  major 0.149, employer 0.090, employer_city 0.076.
- Recall, held-out template 5: 0.000. Fact loss: 0.692 (training templates), 2.227 (held-out).
- Set B (8,000 people never trained on): recall 0.0047 training templates, 0.000 held-out;
  fact loss 1.130 and 2.479.

Measurement check (no evaluation code changed): on 600 training-template prompts, real greedy
decoding gives exact-match recall 0.060, identical to the teacher-forced recall the evaluation uses
on the same prompts. Greedy outputs have the right form but the wrong value, e.g.
"Nosti Krefluthgi was born on" -> " August 27, 1943." (expected " February 7, 1984.").

Decision table row that applies: **"Gate 2 failed: models undertrained or broken. Fix training; no
analysis is valid yet."**
