# Experiment F: person assembly and transcoders — report (Tasks 1 and 1b only)

Plan: `binding/PLAN.md` including Revision 1. Toy models of Phase 2 (6 layers, d = 256). Seed 0
everywhere. Pilot `P2-main-2k-s0` numbers were used for development only and are not results.

**Scope.** Tasks 1 and 1b were run on all 8 runs (main-2k, main-8k, main-32k seeds 0 and 1,
ctrl-8k-x4 s0, fork-A s0). Task 2 stopped at gate F1 on the pilot: T1, T2 and, after Revision 1,
T3 all failed (T3 on the dead-unit criterion on every one of layers 1–5). Per the plan, no other
recipe may be tried, so **no transcoder was trained on any real model and Tasks 3 and 4 were not
run.** Every row that needs transcoders is marked *not measured*. Numbers and a diagnostic of the
failure are in [`binding/STOP.md`](../../binding/STOP.md).

## Task 1: interaction at the last-name position

Contexts START (`<eot>F L`) and MID (`<eot>The employer of F L`); 500 real and 500 control
quadruples per run; IR = ‖I‖ / D at 13 points; difference = median real IR − median control IR,
bootstrap 95% CI (2,000 resamples). Per-point values: `task1/summary.csv`; plots
`task1/ir_START.png`, `task1/ir_MID.png`.

| Run | Routing heads | START: P_bind | START: max diff [CI] | MID: P_bind | MID: max diff [CI] | Median IR real / control at m5 (START) |
| --- | --- | --- | --- | --- | --- | --- |
| main-2k-s0 | 12 | none | 0.009 @a0 [0.001, 0.018] | none | 0.012 @a0 [0.006, 0.025] | 0.36 / 0.36 |
| main-2k-s1 | 13 | none | 0.008 @a5 [0.001, 0.014] | none | 0.009 @m5 [−0.001, 0.017] | 0.34 / 0.34 |
| main-8k-s0 | 6 | none | 0.003 @a4 [−0.005, 0.009] | none | 0.004 @a0 [−0.007, 0.014] | 0.37 / 0.37 |
| main-8k-s1 | 11 | none | 0.004 @m3 [−0.003, 0.008] | none | 0.008 @m5 [0.001, 0.019] | 0.34 / 0.34 |
| main-32k-s0 | 4 | none | 0.012 @a1 [0.004, 0.018] | none | 0.007 @m5 [−0.003, 0.017] | 0.23 / 0.22 |
| main-32k-s1 | 3 | none | 0.004 @a1 [−0.005, 0.010] | none | 0.001 @a3 [−0.009, 0.009] | 0.27 / 0.28 |
| ctrl-8k-x4-s0 | 24 | none | 0.137 @m5 [−0.011, 0.251] | none | 0.053 @a3 [−0.036, 0.127] | 1.05 / 0.91 |
| fork-A-s0 | 11 | none | 0.004 @a5 [−0.005, 0.010] | none | 0.005 @a3 [−0.002, 0.011] | 0.45 / 0.45 |

- Median real IR after layer 0's attention (a0) is 0.10–0.17 in START (0.02–0.03 in fork-A) and
  rises to 0.23–0.45 at m5 (1.05 in ctrl-8k-x4); control IR is the same within 0.01 except
  ctrl-8k-x4.
- Routing-head ablation (step 2): no P_bind, so it is reported at m5 only; the ablated real −
  control difference at m5 is −0.033 to 0.010 in all runs and contexts.
- Step 3 (MLP ablation at P_bind) did not apply: P_bind is none everywhere.
- Binding neurons (step 4): per real quadruple 10.6–13.9 per layer in the main runs, 13.1–16.6 in
  ctrl-8k-x4, against a control count of 10.23–10.24. The largest share of the 500 people that any
  one neuron is binding for is 0.042–0.056.

## Task 1b: where a known person is told apart

END prompts: CITY `<eot>F L was born in`, EMPLOYER `<eot>The employer of F L is`. Outputs:
`task1b/<run>/metrics.json`, heat maps `task1b/<run>/trace_*.png`, `task1b/summary.csv`,
`task1b/interaction_summary.csv`, `task1b/ir_END_*.png`.

**Scoring (implementation choice, stated in `binding/task1b.py`).** Every answer begins with the
bare space token, the same for all people. "The person's correct first answer token" was taken
as the first token of the attribute value, read at the answer-space position (prompt + space,
column `SP`), where the model predicts it. Patched positions: first name to END, plus `SP`.
Interaction at `SP` is reported as supplementary (contexts `*_SP`) and used by no decision row.

**Interaction at END.** No run has `P_bind_END` in either prompt; the largest real − control
difference is 0.012 (main-2k-s0 EMPLOYER), and is at most 0.012 in all 16 run × prompt cases.

**Causal tracing**, people 0–199, corrupted = the person's F2 L2 (unknown pair). p_clean
0.88–0.99, p_corrupt ≤ 0.011; 199–200 of 200 items kept (gap ≥ 0.1) in every case.

| Run | MLP-only max score CITY / EMPLOYER | Location | Max MLP score at name positions | Max MLP score at END | Real vs control Δp, SP mlp0 (CITY) | Residual score at SP, a0 (CITY) |
| --- | --- | --- | --- | --- | --- | --- |
| main-2k-s0 | 0.25 / 0.28 | SP, mlp0 | ≤ 0.009 | ≈ 0 | 0.23 / 0.15 | 0.31 |
| main-2k-s1 | 0.30 / 0.29 | SP, mlp0 | ≤ 0.011 | ≈ 0 | 0.27 / 0.17 | 0.43 |
| main-8k-s0 | 0.35 / 0.44 | SP, mlp0 | ≤ 0.005 | ≈ 0 | 0.32 / 0.13 | 0.58 |
| main-8k-s1 | 0.23 / 0.31 | SP, mlp0 | ≤ 0.002 | ≈ 0 | 0.21 / 0.10 | 0.41 |
| main-32k-s0 | 0.14 / 0.15 | SP, mlp0 | ≤ 0.005 | ≈ 0 | 0.13 / 0.07 | 0.92 |
| main-32k-s1 | 0.10 / 0.11 | SP, mlp5 (CITY) / mlp0 (EMPLOYER) | ≤ 0.005 | ≈ 0 | 0.08 / 0.05 | 0.92 |
| ctrl-8k-x4-s0 | 0.53 / 0.56 | SP, mlp0 | ≤ 0.002 | ≈ 0 | 0.53 / 0.20 | 0.86 |
| fork-A-s0 | 0.10 / 0.05 | SP, mlp5 | ≤ 0.015 | ≈ 0 | 0.00 / 0.00 | 0.00 (0.67 at a4) |

- Residual patching at any single name position restores at most 0.06 (the last-name token
  embedding); residual patching at END restores ≈ 0.
- Control (clean = a second unknown pair, target = its argmax token): p_clean of that token
  0.41–0.52 on the pilot; Δp at SP mlp0 is 0.05–0.20, below the real Δp in every run.

## Decision table

| Result | What it means | Applies? |
| --- | --- | --- |
| Gate F1 fails on the pilot after the ladder | Stop; Tasks 3 and 4 are void | **Yes** (T1, T2; and T3 after Revision 1 — `binding/STOP.md`) |
| `P_bind` is m0 in every run, and ablating routing heads cuts the real − control difference at `P_bind` by at least 75% | A person is assembled by layer 0 | No (P_bind none in every run and context) |
| `P_bind` is later than m0 in some runs | Assembly continues past layer 0 | No (P_bind none) |
| Binding neurons per real quadruple at least 3 times the control count in the binding layer, and no neuron is binding for more than 50% of people | Spread of conjunction neurons | No (no binding layer; at most 16.6 vs 10.24, i.e. < 1.7× in every layer) |
| PERSON count on trained people at least 5 times the unseen null in some layer, for every real run | Person-specific units | Not measured (no transcoders) |
| That PERSON count rises at least 2× from main-2k to main-8k and main-8k to main-32k, ctrl-8k-x4 within 25% of main-8k | A table | Not measured (no transcoders) |
| PERSON count on trained people below 2 times the unseen null in every layer | Shared code, not a table | Not measured (no transcoders) |
| STORY-unit recurrence main-2k vs main-32k ≥ 80% of the ceiling, untrained null < 30% of it | Shared building blocks | Not measured (no transcoders) |
| Median AND index of PERSON units at least 0.2 above that of STORY units | Person units are conjunctions | Not measured (no transcoders) |
| Real − control at END reaches 0.1 with CI above 0 at some point (`P_bind_END` exists) | Told apart at the answer position | No (max 0.012) |
| MLP-only tracing: largest score at END in a layer ≥ 1, name-position scores below half of it | Facts looked up when asked for | No (largest score is at SP, not END; END ≈ 0) |
| MLP-only tracing: largest score at the last-name position | Facts attached at the name | No (≤ 0.015 at name positions) |
| T3 passes F1 on layers 1–5 | Tasks 3 and 4 valid for those layers | No (`binding/STOP.md`) |

Not covered by a row: in all 16 run × prompt cases the largest MLP-only score sits at the
answer-space position SP, one token after END — in the layer-0 MLP in 13 of 16 cases, the layer-5
MLP in the other 3 (fork-A both prompts, main-32k-s1 CITY).

## Not run

- Task 2 on real models, Task 3, Task 4: void after the gate F1 stop. The scripts
  (`binding/task2.py`, `binding/task3.py`, `binding/task4.py`) are written and revised for
  Revision 1; `task3.py` and `task4.py` have never been run.
