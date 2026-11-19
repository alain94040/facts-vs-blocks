# Circuit probe: can Qwen run a small program in its head?

Script: `circuit/probe/probe.py` (seed 0, bfloat16 on MPS, plain text, no chat template). Per-model outputs in
`results/circuit/probe/<model>/` (`metrics.json`, `cmd.txt`, `differential.csv`). Correct answer to the exact program: **7**.

Choices made where PROBE.md left room:
- 3-shot examples (same variables and operators `+ * -`): `a=3,b=a+1,a=b*3,c=a-b` → 8; `a=2,b=a+1,a=b*2` → 3; `a=1,b=a+4,a=b*2` → 5.
- Random programs keep the exact shape: `a = L1; b = a op L2; a = b op L3; c = a op b; print(c)`, with each op drawn from `+ - *`.
  Line 3 overwrites `a` and line 4 reads it. Rejection sampling enforced literals 0..9, intermediates 0..20, answer 0..9 and answer ∉ literals.
  Answer distribution (seed 0): 0:17, 1:1, 2:2, 4:3, 5:4, 6:3, 7:4, 8:11, 9:5
- The baselines read digits from the full 3-shot prompt. Ties for most frequent digit go to the first one seen.
- Seconds per prompt = one forward pass, which excludes the warm-up.
- p(true) is the true digit's probability when it is among the top 5, and 0 otherwise.

## Qwen/Qwen2.5-7B-Instruct

**Exact program, top 5**

- zero-shot: `6` 0.354 `2` 0.215 `3` 0.101 `4` 0.101 `1` 0.089
- 3-shot: `1` 0.449 `9` 0.350 `7` 0.165 `8` 0.017 `5` 0.014

**Differential table (3-shot)**: 2/13 correct; 13 variants skipped (answer <0 or >9: (a) 4->7=10, (a) 4->8=11, (a) 4->9=12, (b) 3->6=10, (b) 3->7=11, (b) 3->8=12, (b) 3->9=13, (c) 2->3=14, (c) 2->4=21, (d) line2 +->*=12, (d) line3 *->-=-2, (d) line4 -->+=21, (d) line4 -->*=98).

| variant | true | top token | p(top) | correct | p(true) |
|---|---|---|---|---|---|
| (a) 4->1 | 4 | 5 | 0.518 | no | 0.016 |
| (a) 4->2 | 5 | 5 | 0.458 | yes | 0.458 |
| (a) 4->3 | 6 | 7 | 0.787 | no | 0.000 |
| (a) 4->5 | 8 | 1 | 0.701 | no | 0.007 |
| (a) 4->6 | 9 | 1 | 0.866 | no | 0.081 |
| (b) 3->1 | 5 | 7 | 0.488 | no | 0.261 |
| (b) 3->2 | 6 | 1 | 0.422 | no | 0.329 |
| (b) 3->4 | 8 | 1 | 0.785 | no | 0.024 |
| (b) 3->5 | 9 | 1 | 0.456 | no | 0.456 |
| (c) 2->1 | 0 | 1 | 0.539 | no | 0.288 |
| (d) line2 +->- | 1 | 1 | 0.672 | yes | 0.672 |
| (d) line3 *->+ | 2 | 3 | 0.582 | no | 0.311 |
| (e) no overwrite (d = b * 2; c = d - b) | 7 | 1 | 0.409 | no | 0.281 |

**50 random programs (3-shot)**: accuracy 0.44 (22/50). Answer 0: 11/17; answers 1-9: 11/33.
Top-token counts: `0`:14, `1`:11, `7`:5, `5`:4, `8`:4, `4`:4, `6`:3, `2`:3, `3`:2.

Baselines: last digit in prompt 0.00; most frequent digit 0.00. The last-digit baseline is 0 by construction, because the last digit is always L3 and the answer is never a literal.

**Seconds per prompt**: 0.073 (65 prompts).

## Qwen/Qwen3-4B

**Exact program, top 5**

- zero-shot: `1` 0.649 `6` 0.128 `2` 0.088 `4` 0.068 `8` 0.029
- 3-shot: `1` 0.846 `2` 0.061 `7` 0.042 `8` 0.012 `9` 0.012

**Differential table (3-shot)**: 2/13 correct; 13 variants skipped (answer <0 or >9: (a) 4->7=10, (a) 4->8=11, (a) 4->9=12, (b) 3->6=10, (b) 3->7=11, (b) 3->8=12, (b) 3->9=13, (c) 2->3=14, (c) 2->4=21, (d) line2 +->*=12, (d) line3 *->-=-2, (d) line4 -->+=21, (d) line4 -->*=98).

| variant | true | top token | p(top) | correct | p(true) |
|---|---|---|---|---|---|
| (a) 4->1 | 4 | 4 | 0.542 | yes | 0.542 |
| (a) 4->2 | 5 | 6 | 0.247 | no | 0.132 |
| (a) 4->3 | 6 | 1 | 0.508 | no | 0.145 |
| (a) 4->5 | 8 | 1 | 0.771 | no | 0.013 |
| (a) 4->6 | 9 | 1 | 0.451 | no | 0.000 |
| (b) 3->1 | 5 | 1 | 0.319 | no | 0.000 |
| (b) 3->2 | 6 | 1 | 0.773 | no | 0.056 |
| (b) 3->4 | 8 | 1 | 0.601 | no | 0.092 |
| (b) 3->5 | 9 | 1 | 0.662 | no | 0.054 |
| (c) 2->1 | 0 | 7 | 0.473 | no | 0.000 |
| (d) line2 +->- | 1 | 2 | 0.421 | no | 0.226 |
| (d) line3 *->+ | 2 | 2 | 0.598 | yes | 0.598 |
| (e) no overwrite (d = b * 2; c = d - b) | 7 | 1 | 0.857 | no | 0.029 |

**50 random programs (3-shot)**: accuracy 0.34 (17/50). Answer 0: 9/17; answers 1-9: 8/33.
Top-token counts: `1`:26, `0`:12, `7`:4, `4`:3, `8`:2, `3`:1, `5`:1, `2`:1.

Baselines: last digit in prompt 0.00; most frequent digit 0.00. The last-digit baseline is 0 by construction, because the last digit is always L3 and the answer is never a literal.

**Seconds per prompt**: 0.045 (65 prompts).
