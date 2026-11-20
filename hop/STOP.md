# STOP: gate H2 fails, the country swap does not flip HOP2 answers (hop task 3, plan revision 1)

Gate H2 (hop/PLAN.md) requires a TEST HOP2 flip rate ≥ 0.5 at `s*` and a random-inject flip rate
≤ 0.1. The random-inject part passes (0.005). **The swap flip rate is 0.064 at `s* = 8`.** By the
plan, country features found by REF selectivity do not move the answer even when the country is
written in the prompt, so the swap on TWO prompts cannot be read. Tasks 4–6 were not run.

Numbers are in `results/hop/task3/` (`metrics.json`, `pairs.csv`, `swaps.json`).

## Earlier gates and tasks

- **H0 passed** (`results/hop/task0/metrics.json`).
- **H1 passes under revision 1** (country-rank rule for HOP1; `results/hop/task1/`). TARGET is
  eligible: ` Mexico` ranks first among country tokens after its HOP1 prompt, although top-1 is
  ` the`. Eligible items: DESC 8 of 12, CAP 83 of 88.
  - The CAP split (seed 0) gives 10 DEV items: Libya, France, Nicaragua, Saint Lucia, Oman, Jordan,
    Ukraine, Ghana, Mauritania, Gabon. TEST has 73 items.
- **Task 2** (`results/hop/task2/`):
  - 4,585 country features. Mexico has 39 and Brazil 25.
  - M(TARGET) is one feature: layer 4, feature 67, at ` States`.
  - Median |M| is 5 on DESC and 7 on CAP, where 5 of the 83 items have an empty M.

## Scale on DEV (10 items × 3 swap countries = 30 pairs per scale)

| s | Flips | Flip rate | Mean change in logp(answer of C') − logp(answer of C) | Pairs with top-1 ` the` |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.033 | +2.32 | 0 |
| 2 | 2 | 0.067 | +4.11 | 1 |
| 4 | 3 | 0.100 | +6.67 | 2 |
| 8 | 2 | 0.067 | +9.55 | 10 |

No scale reaches 0.5, so `s* = 8` by the plan's rule.

## Control on TEST (73 items × 3 swap countries = 219 pairs, s = 8)

| Condition | Flip rate | Mean change in logp(answer of C') − logp(answer of C) | Most frequent top-1 tokens |
| --- | --- | --- | --- |
| swap | 0.064 (14 / 219) | +13.98 | ` the` 69, ` English` 16, ` ` 13, `,` 9, ` Spanish` 9 |
| random_inject | 0.005 (1 / 219) | +8.11 | ` the` 27, ` is` 21, ` English` 12, `,` 11, ` in` 11 |
| suppress | 0.000 | +0.95 | ` Spanish` 36, ` English` 33, ` Arabic` 24, ` Portuguese` 12, ` German` 9 |

Every pair had at least one suppressed and one injected feature. The medians per pair are 26
suppression interventions (feature × position) and 23 injected features.

## Implementation, for checking

- Interventions use `ReplacementModel.feature_intervention` with library defaults
  (`freeze_attention=True`, no constrained layers).
- P is the country token and every later position.
- **Suppress:** each country feature of C that is active at a position in P is set to 0 there.
- **Inject:** each country feature of C' within ±1 layer of a layer holding a country feature of C
  is set to `s × m_C'(l, i)` at every position in P.
- **Code:** `hop/swap.py` and `hop/task3.py`.

The scale was not tuned beyond the plan's {1, 2, 4, 8}, and no other setting was tried.

## Note on an earlier run

Before revision 1 was applied, Tasks 1–3 ran with the old HOP1 top-1 rule, with TARGET added by
hand (commits `0789a16`, `7274a2f`, `ff31e31`). Those outputs are replaced by this run. They gave the
same gate H2 outcome: TEST swap flip rate 0.077 at `s* = 8`.
