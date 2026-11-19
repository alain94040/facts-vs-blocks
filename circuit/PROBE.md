# Quick probe: can Qwen run a small program in its head?

For the local agent in the facts-vs-blocks repository. This is a short probe, run before Experiment G
(`circuit/PLAN.md`). The `CLAUDE.md` rules apply.
First run `ps aux | grep -i python` and stop if another experiment is using the GPU.

## Question

Does the model answer this program instantly, as its single next token, with no written reasoning?

```
a = 4
b = a + 3
a = b * 2
c = a - b
print(c)
```

The correct answer is 7.

## Set-up

- Models, in this order:
  1. `Qwen/Qwen2.5-7B-Instruct`, already cached under `data/cache/hf` from Experiment C. Use the existing environment with `transformers` (`repeat/requirements.txt`).
  2. `Qwen/Qwen3-4B`. Check `df -h` first, then download it to `data/cache/hf`.
- Run in float16 or bfloat16 on MPS.
- Use plain text: no chat template, no thinking mode.
- Put the script in `circuit/probe/probe.py` and the outputs in `results/circuit/probe/`
  (`metrics.json`, `cmd.txt`, and `differential.csv`).
- Prompt format: the program, then a line `# output: `. The answer is the next token (Qwen gives each digit its own token). For every prompt, record the top 5 next tokens with their probabilities.

## Steps

1. **The exact program above**, asked two ways:
   - zero-shot;
   - 3-shot: three worked examples of the same shape with different values, each followed by `# output: <answer>` and a blank line.
2. **Differential table**, 3-shot. Change one thing at a time in the program above. For each variant, record the true answer, the model's top token and its probability.
   - (a) the literal 4 → each of 1..9;
   - (b) the literal 3 → each of 1..9;
   - (c) the literal 2 → each of 1..4;
   - (d) each operator → each of the other two of `+ - *`;
   - (e) no overwrite: the third line becomes `d = b * 2` and the fourth `c = d - b`.

   Skip any variant whose true answer is negative or above 9, and report how many were skipped.
3. **Accuracy on 50 random programs** of the same shape, seed 0, 3-shot. Each program has:
   - 4 assignment lines, one of which overwrites a variable;
   - a last line that reads the overwritten variable;
   - literals 0..9 and intermediate values 0..20;
   - an answer 0..9 that is not equal to any literal in the program.
4. **Shortcut baselines** on the same 50 programs, with no model:
   - answer = the last digit in the prompt;
   - answer = the most frequent digit.

## Report

Write `results/circuit/probe/REPORT.md`, commit the script and `results/circuit/probe/` with the
message `circuit probe: <what>`, and push. Do not commit model weights. The report holds a compact
summary per model:
- the exact program's top 5 tokens, zero-shot and 3-shot;
- the differential table (variant, true answer, top token, probability, correct?);
- accuracy on the 50 programs, and the two baselines;
- seconds per prompt.
