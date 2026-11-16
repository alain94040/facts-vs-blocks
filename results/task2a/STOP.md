# STOP: Task 2a pilot ladder exhausted (P1 and P2 both fail)

PLAN.md Task 2a: "If P2 also fails, write STOP.md with both sets of numbers and stop." No Task 2
run was started; no recipe beyond P1 and P2 was tried.

Change 1 (single-token entities) is done: 1,550 entity tokens (ids 4096-5645), 28,750 template
checks pass, story streams bit-identical to Task 1, mean biography 77.2 tokens (was 117.3).
See `results/task2a/prepare/metrics.json` and `results/task2a/samples.txt`.

## Pilots (main-2k, seed 0, everything else as in Task 2)

| | P1 (100 exposures) | P2 (300 exposures) | Pass threshold |
| --- | --- | --- | --- |
| Stream tokens / steps | 315.6M / 4,797 | 346.9M / 5,272 | |
| Wall time, tokens/s | 38.9 min, 135.6k | 42.7 min, 135.9k | |
| **Recall, training templates (set A, 2,000 people)** | **0.643** (fail) | **0.955** (pass) | >= 0.80 |
| **Name-copy accuracy (5,000 mentions, 1,000 unseen people)** | **0.971** (pass) | **0.870** (fail) | >= 0.90 |
| Pilot result | FAIL | FAIL | both |
| First-name position predicts the bare space | 0.997 | 0.994 | not gated |
| Story val loss change 90% -> 100% | 0.0119 | 0.0115 | (< 0.02 in Gate 2) |
| Story val loss, full split | 1.6938 | 1.6887 | |
| Recall, held-out template 5 (set A) | 0.067 | 0.186 | not gated |
| Fact loss, training / held-out templates (set A) | 0.388 / 1.439 | 0.058 / 1.375 | |
| Recall, training / held-out (set B, 8,000 unseen) | 0.0047 / 0.0005 | 0.0043 / 0.0009 | |
| Fact loss, training / held-out (set B) | 1.944 / 2.572 | 2.640 / 3.218 | |
| Random-token induction ratio | 0.908 | 0.925 | not gated |

Recall per attribute, training templates:

| | birth_date | birth_city | university | major | employer | employer_city |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | 0.044 | 0.900 | 0.718 | 0.318 | 0.977 | 0.903 |
| P2 | 0.838 | 0.993 | 0.948 | 0.958 | 0.999 | 0.994 |

## Measurement check on name-copy (read-only, no evaluation code changed)

Recomputed one biography at a time; numbers match the evaluation exactly (P1 146 errors,
P2 651 errors, of 5,000). Kind of wrong top prediction at the scored position:

| | Last name of a person in the trained main-2k set | Any other token |
| --- | --- | --- |
| P1 | 71 | 75 |
| P2 | 591 | 60 |

Accuracy by mention (2nd to 6th): P1 0.965, 0.972, 0.974, 0.972, 0.971;
P2 0.861, 0.869, 0.877, 0.873, 0.869.

Files: `results/task2a/P1-main-2k-s0/`, `results/task2a/P2-main-2k-s0/`,
`results/task2a/pilot_check_*/`. Decision table row that applies: **"Gate 2 failed: models
undertrained or broken. Fix training; no analysis is valid yet."** Per CLAUDE.md rule 11, the fix
is a PLAN.md change by the reviewer.
