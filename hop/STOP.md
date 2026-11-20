# STOP: gate H2 fails, the country swap does not flip HOP2 answers (hop task 3)

Gate H2 (hop/PLAN.md) requires a TEST HOP2 flip rate ≥ 0.5 at `s*` and a random-inject flip rate
≤ 0.1. The random-inject part passes (0.0). **The swap flip rate is 0.077 at `s* = 8`.** By the
plan, country features found by REF selectivity do not move the answer even when the country is
written in the prompt, so the swap on TWO prompts cannot be read. Tasks 4–6 were not run.

Numbers are in `results/hop/task3/` (`metrics.json`, `pairs.csv`, `swaps.json`).

## Earlier gates

- **H0 passed** (`results/hop/task0/metrics.json`).
- **H1 failed** because TARGET's HOP1 top-1 is ` the`. On 2026-11-20 the reviewer decided to keep
  TARGET and continue (`results/hop/task1/metrics.json`, key `reviewer_include`).
- **Task 2** found 4,585 country features. Mexico has 39 and Brazil 25. M(TARGET) is one feature:
  layer 4, feature 67, at ` States` (`results/hop/task2/`).

## Scale on DEV (10 items × 3 swap countries = 30 pairs per scale)

| s | Flips | Flip rate | Mean change in logp(answer of C') − logp(answer of C) | Pairs with top-1 ` the` |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0.000 | +3.47 | 3 |
| 2 | 5 | 0.167 | +5.94 | 2 |
| 4 | 9 | 0.300 | +8.90 | 5 |
| 8 | 1 | 0.033 | +11.38 | 9 |

No scale reaches 0.5, so `s* = 8` by the plan's rule.

## Control on TEST (69 items × 3 swap countries = 207 pairs, s = 8)

| Condition | Flip rate | Mean change in logp(answer of C') − logp(answer of C) | Most frequent top-1 tokens |
| --- | --- | --- | --- |
| swap | 0.077 (16 / 207) | +13.32 | ` the` 63, ` English` 14, ` Arabic` 13, ` Spanish` 11, ` is` 8 |
| random_inject | 0.000 | +7.22 | ` the` 33, ` is` 16, `,` 13, ` ` 11, ` Arabic` 9 |
| suppress | 0.000 | +0.84 | ` Spanish` 36, ` Arabic` 33, ` English` 27, ` Portuguese` 12, ` French` 9 |

Every pair had at least one suppressed and one injected feature. The medians per pair are 26
suppression interventions (feature × position) and 20 injected features.

## Implementation, for checking

- Interventions use `ReplacementModel.feature_intervention` with library defaults
  (`freeze_attention=True`, no constrained layers).
- P is the country token and every later position.
- **Suppress:** each country feature of C that is active at a position in P is set to 0 there.
- **Inject:** each country feature of C' within ±1 layer of a layer holding a country feature of C
  is set to `s × m_C'(l, i)` at every position in P.
- **Code:** `hop/swap.py` and `hop/task3.py`.

The scale was not tuned beyond the plan's {1, 2, 4, 8}, and no other setting was tried.
