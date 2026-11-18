# STOP: gate F1 fails on the pilot after the ladder (binding task 2)

Pilot model `P2-main-2k-s0`, transcoder seed 0. Per binding/PLAN.md Task 2, both declared recipes
were trained and both fail gate F1. Decision table row: **"Gate F1 fails on the pilot after the
ladder: Stop; Tasks 3 and 4 are void."** No transcoders were trained on real models; tasks 3 and 4
were not run (their scripts, `binding/task3.py` and `binding/task4.py`, are written but unrun).

Gate F1, for each layer replaced alone: FVU ≤ 0.15; FACT recall ≥ 0.85 × unedited (0.85 × 0.9542 =
0.811); STORY loss up by ≤ 0.10 nats (≤ 1.7906); dead units ≤ 30%.

Unedited model: FACT recall 0.9542, STORY loss 1.6906.

## T1: 16,384 units, k = 32, 20M positions

| Layer | FVU | Dead | Recall (layer replaced) | STORY loss | Fails on |
| --- | --- | --- | --- | --- | --- |
| L0 | 0.096 | 0.055 | 0.564 | 1.853 | recall, loss |
| L1 | 0.201 | 0.181 | 0.881 | 1.729 | FVU |
| L2 | 0.216 | 0.181 | 0.861 | 1.733 | FVU |
| L3 | 0.258 | 0.129 | 0.832 | 1.745 | FVU |
| L4 | 0.188 | 0.108 | 0.828 | 1.754 | FVU |
| L5 | 0.106 | 0.059 | 0.822 | 1.773 | — (passes) |

All six replaced: recall 0.215, STORY loss 2.299.

## T2: 32,768 units, k = 64, 40M positions

| Layer | FVU | Dead | Recall (layer replaced) | STORY loss | Fails on |
| --- | --- | --- | --- | --- | --- |
| L0 | 0.057 | 0.363 | 0.727 | 1.768 | recall, dead |
| L1 | 0.090 | 0.614 | 0.927 | 1.707 | dead |
| L2 | 0.094 | 0.633 | 0.925 | 1.708 | dead |
| L3 | 0.103 | 0.611 | 0.917 | 1.711 | dead |
| L4 | 0.077 | 0.610 | 0.914 | 1.714 | dead |
| L5 | 0.052 | 0.496 | 0.904 | 1.727 | dead |

All six replaced: recall 0.515, STORY loss 1.923.

Dead = never among the active top-k units on 1,000,192 held-out STORY positions nor on the FACT
prompts. (STORY-only dead fractions, T2: 0.383, 0.616, 0.635, 0.616, 0.619, 0.549.)

T2 passes FVU, recall (L1–L5) and STORY loss on every layer; it fails the dead-unit criterion on every
layer and recall at L0. T1 fails FVU on L1–L4 and recall/loss at L0.

## Files

- `results/binding/task2/P2-main-2k-s0/metrics_T1.json`, `train_tc0_T1.json`, `cmd_*_T1.txt`
- `results/binding/task2/P2-main-2k-s0/metrics_T2.json`, `train_tc0_T2.json`, `cmd_*_T2.txt`
  (`metrics.json` / `train_tc0.json` are the same T2 files at the plan's output path)
- Training logs: `logs/binding_tc_pilot*.log`
- Checkpoints (not committed): `checkpoints/binding/P2-main-2k-s0/tc0_T1/`, `.../tc0/` (T2)

Task 1 results (committed in 09364b9) stand; they do not use transcoders.
