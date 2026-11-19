# STOP: gate F1 fails on the pilot with recipe T3 (binding task 2, Revision 1)

Pilot model `P2-main-2k-s0`, transcoder seed 0. Revision 1 of binding/PLAN.md: "Run T3 on the
pilot. If T3 fails F1 on any of layers 1–5, write `STOP.md` and stop; no other recipe may be
tried." T3 fails F1 on all of layers 1–5, on the dead-unit criterion. No transcoder was trained
on any other model; Tasks 3 and 4 were not run (scripts `binding/task3.py` and `binding/task4.py`
are revised for Revision 1 but unrun). Task 1 and Task 1b results stand (no transcoders).

Gate F1 (layers 1–5; layer 0 reported and flagged), each layer replaced alone: FVU ≤ 0.15; FACT
recall ≥ 0.85 × unedited (≥ 0.811); STORY loss ≤ unedited + 0.10 (≤ 1.7906); dead units ≤ 30%.
Unedited: FACT recall 0.9542, STORY loss 1.6906.

## T3: 32,768 units, k = 64, 40M positions, auxiliary loss (top 512 dead, coefficient 1/32)

| Layer | FVU | Dead | Recall | STORY loss | Fails on |
| --- | --- | --- | --- | --- | --- |
| L0 (flagged) | 0.057 | 0.358 | 0.726 | 1.771 | recall, dead |
| L1 | 0.088 | 0.615 | 0.927 | 1.707 | dead |
| L2 | 0.094 | 0.629 | 0.926 | 1.708 | dead |
| L3 | 0.104 | 0.604 | 0.915 | 1.711 | dead |
| L4 | 0.078 | 0.607 | 0.912 | 1.714 | dead |
| L5 | 0.052 | 0.501 | 0.904 | 1.727 | dead |

All six replaced: recall 0.502, STORY loss 1.926.

For comparison, T2 (same without the auxiliary loss): dead 0.363, 0.614, 0.633, 0.611, 0.610,
0.496; FVU 0.057–0.103; recall 0.727 (L0), 0.904–0.927 (L1–L5). T1: see the history of this file
(commit e2b728f); FVU 0.20–0.26 on L1–L4.

## Diagnostic (measured after the failure; changes no gate, no retry)

`binding/diag_dead.py`, `results/binding/diag_dead/P2-main-2k-s0/metrics.json`.

Units dead by the training criterion (no activation in the last 10M training positions) at the
end of T3 training: 56, 3,117, 2,998, 1,956, 1,742, 510 of 32,768 (0.2%–9.5%). Dead by the gate
(no activation on 1,000,192 held-out STORY positions + 317,220 FACT-prompt positions): 36%–63%.

| Units by activation count on the gate data | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.358 | 0.615 | 0.629 | 0.604 | 0.607 | 0.501 |
| 1 | 0.218 | 0.179 | 0.183 | 0.199 | 0.202 | 0.221 |
| 2–3 | 0.196 | 0.091 | 0.088 | 0.097 | 0.096 | 0.126 |
| 4–10 | 0.123 | 0.042 | 0.038 | 0.035 | 0.027 | 0.049 |
| 11–100 | 0.050 | 0.019 | 0.013 | 0.014 | 0.012 | 0.033 |
| 101–1,000 | 0.021 | 0.017 | 0.013 | 0.014 | 0.014 | 0.025 |
| > 1,000 | 0.035 | 0.038 | 0.035 | 0.038 | 0.042 | 0.044 |

- Mean active units per position is 61–64 (k = 64), so nearly all activations fall on the ~4%
  of units with more than 1,000 activations.
- Adding held-out biographies of all 2,000 trained people (500,224 positions; people ≥ 1,000 are
  not in the FACT prompts) lowers the gate-dead fraction by at most 2.4 points (L0) and 0.4–2.4
  points elsewhere: units active only on people ≥ 1,000 are 0.1%–1.3%.

## Training speed

T3 on the pilot took 10,318 s (2.9 h): steps 0–6,000 ran at about 320 s per 1,000 steps; after
the auxiliary loss switched on and the dead count grew, steps 8,000–9,000 took 3,147 s, with the
machine at 25 GB of swap. The training code gathered a different-size matrix each step
(`pre[:, dead]`); it now uses fixed shapes (`masked_fill` + top-512), mathematically the same.

Speed check of the fixed version (pilot, 1,000 steps, auxiliary loss forced on from step 20 by
setting the dead window to 20 steps; scratch run, not kept): 1,078 s for 1,000 steps, 95–110 s per
100 steps throughout (about 3,800 positions/s), with no growth over the run. Without the
auxiliary loss the same steps run at about 320 s per 1,000. So the fixed version does not slow
down over time, but the full-width top-512 makes each step about 3.4 times slower: a full T3 run
would take about 2,441 × 0.32 s + 7,324 × 1.08 s ≈ 2.4 h, not 50 min.

## Files

- T1: `results/binding/task2/P2-main-2k-s0/metrics_T1.json`, `train_tc0_T1.json`
- T2: `metrics_T2.json`, `train_tc0_T2.json`
- T3: `metrics_T3.json`, `train_tc0_T3.json` (same as `metrics.json`, `train_tc0.json`); logs `logs/binding_tc_pilot_T3*.log`
- Checkpoints (not committed): `checkpoints/binding/P2-main-2k-s0/tc0_T1/`, `tc0_T2/`, `tc0/` (T3)
