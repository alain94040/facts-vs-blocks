# STOP: gate H1 fails, TARGET is not eligible (hop task 1)

Gate H1 (hop/PLAN.md) requires that TARGET is eligible and that at least 25 CAP items are eligible.
The CAP part passes (79 eligible). **TARGET is not eligible**: its TWO and HOP2 prompts are top-1
correct, but its HOP1 prompt is not. The plan says what to do when TARGET's TWO is wrong (stop) and
when fewer than 25 CAP items are eligible (continue with TARGET and DESC). It does not say what to do
when TWO is right and HOP1 is wrong, so I stopped. Tasks 2–6 were not run. All numbers are in
`results/hop/task1/` (`items.csv`, `metrics.json`).

Gate H0 passed (Task 0, `results/hop/task0/metrics.json`).

## TARGET

| Kind | Prompt | Answer | Top-5 (probability) |
| --- | --- | --- | --- |
| TWO | `The language most commonly spoken in the country south of the United States is` | ` Spanish` | ` Spanish` 0.499, ` English` 0.156, ` ` 0.031, ` the` 0.022, `:` 0.020 |
| HOP1 | `The country south of the United States is` | ` Mexico` | ` the` 0.113, ` a` 0.100, ` Mexico` 0.076, ` called` 0.055, ` known` 0.046 |
| HOP2 | `The language most commonly spoken in Mexico is` | ` Spanish` | top-1 ` Spanish` |

(TWO top-5 from `results/hop/task0/metrics.json`, `step4_graph.logit_probabilities`; the others from
`results/hop/task1/items.csv`.)

## Accuracy (top-1 correct) and eligible items

| Set | n | TWO | HOP1 | HOP2 | Eligible |
| --- | --- | --- | --- | --- | --- |
| TARGET | 1 | 1.00 | 0.00 | 1.00 | 0 |
| DESC | 12 | 0.75 | 0.75 | 1.00 | 7 |
| CAP | 88 | 0.989 | 0.943 | 0.955 | 79 |

DESC items not eligible:
- DESC01, south of Canada → United States: HOP1 top-1 ` the`.
- DESC02, largest in South America → Brazil: TWO top-1 ` Spanish`.
- DESC03, shaped like a boot → Italy: TWO top-1 ` English`, HOP1 top-1 ` a`.
- DESC05, largest in the world → Russia: TWO top-1 ` Chinese`.
- DESC11, island where Tokyo is → Japan: HOP1 top-1 ` called`.

The two other Mexico descriptions that refer to the United States are eligible:
- DESC07, "directly south of the United States": HOP1 ` Mexico` 0.338.
- DESC08, "borders the United States to the south": HOP1 ` Mexico` 0.386.

DESC06 (capital Mexico City) is eligible too.

CAP items not eligible:
- HOP1 top-1 ` called`: Liberia, Saint Kitts and Nevis, Saint Vincent and the Grenadines, Suriname.
- HOP1 top-1 ` Newfoundland`: Antigua and Barbuda.
- HOP2 wrong: Senegal (` W`), Malawi (` Ch`), Botswana (` Sets`), Namibia (` Afrikaans`).

If the run continues, the CAP split (seed 0) is:
- DEV (10): Honduras, Liechtenstein, Jordan, Bulgaria, Norway, Tajikistan, Zambia, Andorra, Denmark, Bhutan.
- TEST: 69 items.

## For the reviewer

The scripts for Tasks 2–5 are written (`hop/task2.py` … `hop/task5.py`) but have not been run. A
decision is needed on TARGET: whether it can go ahead with HOP1 wrong, or what else should happen.
