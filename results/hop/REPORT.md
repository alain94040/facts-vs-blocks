# Experiment H report: does Gemma-2-2B pass through "Mexico"? (plan revision 2)

Model `google/gemma-2-2b`, float32 on MPS, seed 0. Plan: `hop/PLAN.md` (revision 2, section 7).
All numbers are as measured. The readings follow the decision table only.

**Decision-table row (revision 2):** at `l* = 25`, on TARGET + DESC + DESC2, `W = 14` of 16 items
(expected 1.89 by chance), `p < 1e-5` (no simulated draw out of 100,000 reached 14), language hit
rate 0.94. **Reading: Specific bridge.** See the note on `l*` in Task 4b before using it.

## Gates

| Gate | Check | Result |
| --- | --- | --- |
| H0 | Step 2 log-prob difference ≤ 0.01; identity change ≤ 0.001; graph ≤ 20 min; peak ≤ 48 GB | pass. Difference 0.0, change 0.0, graph 14.9 s, peak 28.9 GB while the graph builds (37.4 GB earlier in the run, with the plain TL model also loaded). Completeness 0.930, replacement 0.734, 13,778 active features. |
| H1 | TARGET eligible; ≥ 25 CAP eligible | pass (revision 1 rule). TARGET eligible, 83 CAP. |
| H2 | TEST HOP2 swap flip ≥ 0.5; random inject ≤ 0.1 | **fail**. Swap 0.064 at `s* = 8`; random inject 0.005. Feature swaps were dropped by revision 2. |
| H3 | At `l*`: leave-one-template-out top-1 ≥ 0.5 and within-language ≥ 0.5 | pass. `l* = 25`, top-1 0.872, within-language 0.944 (chance 0.011 and 0.118). |

## Task 1: behaviour (top-1 accuracy; HOP1 also under the country-rank rule)

| Set | Items | TWO | HOP1 top-1 | HOP1 country rank 1 | HOP2 | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| TARGET | 1 | 1.00 | 0.00 | 1.00 | 1.00 | 1 |
| DESC | 12 | 0.75 | 0.75 | 0.83 | 1.00 | 8 |
| CAP | 88 | 0.99 | 0.94 | 0.99 | 0.95 | 83 (10 DEV, 73 TEST) |
| DESC2 | 12 | 0.92 | 0.83 | 1.00 | 1.00 | 11 |

DESC2 keeps 12 of its 14 descriptions: China and India are not in the country table, so they are
dropped. The only DESC2 item that is not eligible is DESC2-00 (`the country where the Aztec empire
was`, Mexico); its TWO top-1 is ` Nah`. Files: `task1/`, `task4b/desc2_items.csv`.

## Task 2 and Task 3: the dropped swap method (record)

- **Country features.** 4,585 pass the REF selectivity rule; Mexico has 39 and Brazil 25.
- **Bridge sets.** `M(TARGET)` is one feature, (4, 67) at ` States`. Median |M| is 5 on DESC and 7
  on CAP; 5 CAP items have an empty M.
- **TARGET graph.** None of the 30 most influential nodes on ` Spanish` is in `M(TARGET)`, and 20 of
  them never fire on any REF sentence.
- **Task 3, DEV flip rate by scale.**

  | s | 1 | 2 | 4 | 8 |
  | --- | --- | --- | --- | --- |
  | Flip rate | 0.033 | 0.067 | 0.100 | 0.067 |

  No scale reaches 0.5, so `s* = 8`.
- **Task 3, TEST at `s* = 8`.** Swap 0.064 (14 of 219 pairs), random inject 0.005, suppress 0.000.
  The mean change in logp(C' answer) − logp(C answer) is +13.98, +8.11 and +0.95. ` the` is top-1
  in 69 of the swap pairs.
- The full record is in `task2/`, `task3/` and git history (the earlier `hop/STOP.md`).

## Task 4b: country probe

**Training prompts.** A prompt is kept when it passes the country-rank rule. Kept: P1 91 of 92,
P2 92 of 92, P3 90 of 92. All 92 countries have a class vector.

**Added before the run.** Capitals for the four hand-added rows: Mexico City, Washington, D.C.,
Ottawa and Moscow. Mexico's class is built from prompts that name "Mexico City", a word cue TARGET
does not have.

**Control, leave one template out** (`task4b/control_by_layer.csv`, `probe_control.png`). 273
held-out prompts; 177 of them have a language shared by ≥ 2 countries.

| Layer | 0 | 4 | 8 | 12 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top-1 | 0.05 | 0.03 | 0.18 | 0.05 | 0.28 | 0.12 | 0.63 | 0.62 | 0.68 | 0.67 | 0.76 | 0.74 | 0.81 | 0.87 |
| Within-language | 0.24 | 0.14 | 0.29 | 0.23 | 0.51 | 0.38 | 0.77 | 0.79 | 0.83 | 0.81 | 0.90 | 0.88 | 0.85 | 0.94 |

`l* = 25`, the last layer, which has the highest within-language accuracy.

**Test at `l*`, last position** (`task4b/test_items.csv`). The decision uses TARGET + DESC + DESC2,
the 16 items whose language has ≥ 2 countries. DESC10, DESC11, DESC2-10 and DESC2-11 are excluded
because Greek, Japanese and Italian each have only one country.

| Set | n | W (within-language rank 1) | Expected W | p | Language hit rate | Bridge top-1 over all countries |
| --- | --- | --- | --- | --- | --- | --- |
| **TARGET + DESC + DESC2** | 16 | **14** | 1.89 | < 1e-5 | 0.94 | 13 |
| TARGET | 1 | 1 | 0.07 | 0.072 | 1.00 | 1 |
| DESC | 6 | 5 | 0.55 | 3e-5 | 1.00 | 5 |
| DESC2 | 9 | 8 | 1.26 | < 1e-5 | 0.89 | 7 |
| CAP TEST (not in the decision) | 43 | 43 | 5.51 | < 1e-5 | 1.00 | 43 |

- **Misses (within-language rank 2):**
  - DESC00, Canada: top-3 United States, Canada, Antigua and Barbuda.
  - DESC2-05, Colombia: top-3 Spain, Colombia, Chile.
- **DESC2-08, Portugal:** within-language rank 1 but rank 2 over all countries, behind Spain.

**At the description's last token** (reported, not used in the decision):
- W = 8 of 16, with expected 1.89 and p = 9e-5.
- The language hit rate is 0.06, and no item has the bridge top-1 over all countries.
- "United States" or "Andorra" is the top-1 country for most items.
- On CAP TEST, W = 37 of 43.

**TARGET panel** (`task4b/target_probe.png`). Mexico's rank among the 92 countries and among the 14
Spanish-speaking countries:

| Layer | 0 | 4 | 8 | 12 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rank, all countries (last) | 2 | 2 | 23 | 12 | 3 | 20 | 2 | 2 | 2 | 2 | 1 | 2 | 1 | 1 |
| Rank within Spanish (last) | 2 | 1 | 3 | 1 | 1 | 11 | 1 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| Rank within Spanish (` States`) | 4 | 1 | 6 | 1 | 1 | 8 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

At the last position, the top-3 is United States, Mexico, Argentina at layers 18–20. It is Mexico,
United States, Spain at layers 22 and 24, and Mexico, Argentina, Spain at layer 25.

**Note on `l*` (a description, not a reading).** `l* = 25` is the final residual stream. At that
layer the training prompts are about to output the country's name, so the class vectors include
each country's output direction. On the TWO prompts, the same layer is about to output the
language. The rule fixes `l*` by the control, and that rule was applied as written.

For the reviewer, W on the main set at the last position, layer by layer, computed from the saved
ranks and not part of the decision:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 4 | 5 | 0 | 1 | 5 | 1 | 0 | 8 | 3 | 1 | 2 | 3 | 3 | 2 | 1 | 2 | 6 | 4 | 15 | 13 | 15 | 14 | 15 | 15 | 14 | 14 |

Expected W is 1.89 at every layer.

## Task 5: logit lens (`task5/`)

L is the first layer whose rank is below 10. Each cell gives the median, then the count of items
that never reach the top 10 ("none").

| Set | n | L_answer, last | L_bridge, last | L_answer, desc token | L_bridge, desc token |
| --- | --- | --- | --- | --- | --- |
| TARGET | 1 | 19 | none | none | none |
| DESC | 8 | 19 (0 none) | 22 (7 none) | — (8 none) | — (8 none) |
| DESC2 | 11 | 19 (0 none) | 22 (8 none) | — (11 none) | 6 (10 none) |
| CAP (all) | 83 | 20 (0 none) | 25 (26 none) | 0 (62 none) | 0 (20 none) |
| CAP TEST | 73 | 20 (0 none) | 25 (21 none) | 0 (53 none) | 0 (18 none) |

**TARGET:** ` Spanish` enters the top 10 at the last position at layer 19. ` Mexico` never enters
the top 10 at either position (`target_logit_lens.png`).

**M features against L_answer:** an active M feature fires before L_answer at the last position in
8 of 8 DESC items, in 67 of the 68 CAP TEST items with a non-empty M (5 have an empty M), and in
TARGET. DESC2 items have no M,
since Task 2 did not cover them.

## Task 5b: last-token patching (`task5b/`, `patching.png`)

Each eligible TWO item gets 3 donors with a different language. The residual stream at the last
position is replaced after layer `l`. The table gives the donor-answer rate:

| Set | Recipients | Layer 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22–25 | L_patch | Task 5 median L_answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TARGET | 1 | 0.00 | 0.67 | 0.67 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **16** | 19 |
| DESC | 8 | 0.12 | 0.38 | 0.42 | 0.92 | 0.96 | 0.96 | 0.96 | 1.00 | 18 | 19 |
| DESC2 | 11 | 0.03 | 0.24 | 0.27 | 0.64 | 0.64 | 0.85 | 0.85 | 1.00 | 18 | 19 |
| DESC + DESC2 | 19 | 0.07 | 0.30 | 0.33 | 0.75 | 0.77 | 0.89 | 0.89 | 1.00 | 18 | — |
| CAP TEST | 73 | 0.02 | 0.06 | 0.06 | 0.51 | 0.68 | 0.89 | 0.89 | 1.00 | 18 | 20 |

- **Earlier layers:** the donor-answer rate is 0 at layers 0–14 in every set. There the top-1 is
  the recipient's own language, with a few "other" tokens.
- **TARGET donors:** DESC2-08 (Portugal), DESC2-11 (Japan) and DESC00 (Canada).
- **Comparison with E:** E reported, for Qwen 7B, no item-specific information at the last token
  until about layer 18–20 (of 28). Here `L_patch` is 18 of 26 for every set except TARGET, which
  is 16. No further reading, as the plan says.

## Deviations and implementation notes

- **Probe capitals.** Capitals for Mexico, the United States, Canada and Russia were added in
  `hop/data.py` (`EXTRA_CAPITALS`) for the probe prompts. The country table had none for these
  rows, and Mexico's class needs one. CAP items are unchanged.
- **DESC2 in Task 5.** Task 5 also runs the eligible DESC2 items, as a separate set with no M, so
  Task 5b can be set against an L_answer for every set. The TARGET, DESC and CAP numbers are
  unaffected.
- **Plots.** `circuit/.venv` has no matplotlib, and the plan forbids changing it. All plots
  therefore come from `hop/plots.py`, run under the repository's `.venv`, reading only the saved
  JSON. The plot code was removed from `task4b.py` and `task5.py`, and Task 4b was rerun: its
  `metrics.json` is identical.
- **Pooled control.** The leave-one-template-out control pools the held-out predictions of the
  three folds. On a tie, `l*` is the earliest layer (there was no tie).
- **Model.** Tasks 4b and 5b use the plain TransformerLens model, loaded as in Task 0 step 2.
  Task 5 uses the ReplacementModel's residual stream, which is identical (Task 0 step 2).
