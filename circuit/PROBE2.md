# Probe 2: which part of the program breaks the single-step answer?

For the local agent in the facts-vs-blocks repository. This follows `circuit/PROBE.md`
(results in `results/circuit/probe/REPORT.md`). The `CLAUDE.md` rules apply.
First run `ps aux | grep -i python` and stop if another experiment is using the GPU.

**Why.** Probe 1 failed on both models. Qwen2.5-7B scored 0.44 on 50 programs and only 11/33 on
answers other than 0; Qwen3-4B scored 0.34 and answered `1` on 26 of 50. The version without an
overwrite failed too. So the failure may come from the chain of three arithmetic steps, not from
tracking variables. This probe separates the two with a ladder of levels, and also checks that the
models solve the original programs when they may write their steps first.

## Set-up

As in Probe 1:
- the same two models, dtype, prompt format (`# output: ` then the answer as the next token) and
  top-5 recording;
- the script in `circuit/probe/probe2.py`, which may import from `circuit/probe/probe.py` without
  changing it;
- outputs in `results/circuit/probe2/<model>/` (`metrics.json`, `cmd.txt`, `levels.csv`).

Every level uses 100 random items (seed 0) and 3-shot prompts. The three worked examples come from
the same level, drawn with seed 1 and different from every test item. Variables are drawn without
replacement from `a b c d e x y z`, so names differ between items. Literals are 0..9, every
intermediate value is 0..20, and every answer is 0..9.

## Levels

`L` is a literal, `u v w` are variables, and `op` is `+` or `-` unless stated otherwise.

| Level | What it tests | Shape (before `print`) | Answer |
| --- | --- | --- | --- |
| C1 | lookup after one overwrite | `u = L1`, `v = L2`, `u = v`, `v = L3`, `print(u)` | L2 |
| C2 | lookup through two copies and two overwrites | `u = L1`, `v = L2`, `w = u`, `u = v`, `v = L3`, `print(w)` or `print(u)` (half each) | L1 or L2 |
| C3 | C2 plus two distractor lines with new variables and literals, inserted at random positions | | as C2 |
| A1 | one operation | `u = L1`, `v = u op L2`, `print(v)` | |
| A2 | two operations in a chain | `u = L1`, `v = u op L2`, `w = v op L3`, `print(w)` | |
| A3 | two operations plus an overwrite | `u = L1`, `v = u op L2`, `u = v op L3`, `print(u)` | |
| A3x | A3 with `op` drawn from `+ - *` | | |
| A4 | the Probe 1 shape | as in Probe 1, with names drawn as here | |

For the C levels the answer is a literal, so record three baselines:
- the last literal;
- the most frequent digit;
- a random literal of the item (expected accuracy reported analytically: the mean of 1 / number of distinct literals).

For the A levels, reject items whose answer equals any literal, as in Probe 1.

## Written steps (one check)

On the 50 random programs of Probe 1 (same seed, same shape), let the model write its steps first.
Each of the three worked examples gets one comment line before its output line, listing the value
after each line in order, for example `# a=3, b=4, a=12, c=8`. The test prompt ends after
`print(c)\n` with no `# output: `. Generate greedily up to 60 tokens and score the first digit after
`# output: ` in the generated text. Report the accuracy, and how many items produced no
`# output:` line.

## Report

Write `results/circuit/probe2/REPORT.md`, commit the script and `results/circuit/probe2/` with the
message `circuit probe2: <what>`, and push. Do not commit model weights. The report contains, per
model:
- a table with one row per level: accuracy, the baselines (C levels), mean p(true), and the three
  most common top tokens with their counts;
- the written-steps accuracy;
- seconds per prompt.
