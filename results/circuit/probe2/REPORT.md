# Circuit probe 2: which part of the program breaks the single-step answer?

Script: `circuit/probe/probe2.py` (imports `probe.py` unchanged; seed 0; bfloat16 on MPS; plain text; 3-shot from the same level, seed 1,
no worked example equal to a test item). Per-model outputs are in `results/circuit/probe2/<model>/` (`metrics.json`, `cmd.txt`, `levels.csv`). 100 items per level.

Choices made where PROBE2.md left room:
- C2/C3: even-indexed items print the copy `w` (answer L1), odd-indexed print `u` (answer L2), so exactly 50/50. Worked examples follow the same parity.
- C3 distractors are `name = L` lines using two more variables from the item's draw and fresh literals 0..9. Each is inserted at a uniform position among the lines before `print`.
- Literals within an item are drawn independently, so they may repeat. That is why the random-literal baseline is above 1/3 in C1.
- The C baselines are computed over the item's own program, not over the worked examples. Ties for most frequent digit go to the first one seen.
- p(true) is the full softmax probability of the true digit's token.
- Written steps: Probe 1's three worked examples, each with a `# a=.., b=.., a=.., c=..` line. Greedy, 60 new tokens.
  Scored on the first digit after the first `# output:`. The first scorer used a regex that could skip a `# output: -k` line.
  I rescored from the saved generations with the corrected function now in the script, and the accuracies did not change.

## Qwen/Qwen2.5-7B-Instruct

| level | accuracy | last literal | most frequent digit | random literal | mean p(true) | top-3 top tokens |
|---|---|---|---|---|---|---|
| C1 | 0.93 | 0.12 | 0.18 | 0.38 | 0.906 | `5`:16 `7`:14 `3`:11 |
| C2 | 0.66 | 0.08 | 0.53 | 0.38 | 0.641 | `5`:17 `3`:15 `7`:11 |
| C3 | 0.60 | 0.10 | 0.39 | 0.25 | 0.595 | `8`:16 `0`:15 `4`:14 |
| A1 | 1.00 | – | – | – | 0.995 | `0`:13 `7`:13 `8`:12 |
| A2 | 0.98 | – | – | – | 0.965 | `9`:15 `7`:12 `4`:12 |
| A3 | 0.97 | – | – | – | 0.964 | `9`:15 `7`:13 `6`:12 |
| A3x | 1.00 | – | – | – | 0.997 | `0`:14 `5`:14 `9`:14 |
| A4 | 0.50 | – | – | – | 0.480 | `0`:32 `4`:12 `1`:12 |

A4 split: answer 0 → 29/36; answers 1–9 → 21/64.

**Written steps** (Probe 1's 50 programs): accuracy 0.86; items with no `# output:` line: 0; items whose output line is negative: 6.

**Seconds per prompt**: 0.052 (single forward pass, levels); 2.04 per written-steps item (generation).

## Qwen/Qwen3-4B

| level | accuracy | last literal | most frequent digit | random literal | mean p(true) | top-3 top tokens |
|---|---|---|---|---|---|---|
| C1 | 0.99 | 0.12 | 0.18 | 0.38 | 0.984 | `5`:15 `7`:13 `3`:13 |
| C2 | 1.00 | 0.08 | 0.53 | 0.38 | 0.999 | `2`:15 `1`:11 `7`:11 |
| C3 | 0.95 | 0.10 | 0.39 | 0.25 | 0.942 | `8`:17 `4`:13 `6`:13 |
| A1 | 1.00 | – | – | – | 1.000 | `0`:13 `7`:13 `8`:12 |
| A2 | 1.00 | – | – | – | 1.000 | `9`:15 `4`:13 `7`:12 |
| A3 | 0.98 | – | – | – | 0.974 | `9`:15 `4`:14 `7`:13 |
| A3x | 0.99 | – | – | – | 0.989 | `9`:15 `0`:14 `5`:14 |
| A4 | 0.44 | – | – | – | 0.381 | `0`:33 `1`:30 `8`:10 |

A4 split: answer 0 → 29/36; answers 1–9 → 15/64.

**Written steps** (Probe 1's 50 programs): accuracy 1.00; items with no `# output:` line: 0; items whose output line is negative: 0.

**Seconds per prompt**: 0.041 (single forward pass, levels); 1.59 per written-steps item (generation).
