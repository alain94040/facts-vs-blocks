# Rules for coding agents in this repository

You are executing `PLAN.md`. Read it fully before writing code. These rules override your defaults.

`docs/BACKGROUND.md` explains the research question, the literature and the reasons behind the design.
Read it when you need context. It is information only: it never changes a task, gate or threshold.

1. Do the tasks in order. Commit after every task with the message `task N: <what>`.
2. Never change an evaluation script, a test set, or a gate threshold after seeing results.
   If a gate fails, stop and write `STOP.md` naming the gate and the numbers.
3. Never retry a failed gate by tuning hyperparameters on evaluation data. Hyperparameters live in `configs/` and are fixed by the plan.
4. Every script takes `--seed` and writes to `results/<task>/<run_name>/` a `metrics.json` plus the exact command in `cmd.txt`.
5. Before any run over 10 minutes, run the same script for 50 steps, confirm loss decreases, and record tokens per second.
6. Keep code plain: PyTorch, numpy, `tokenizers`, `datasets`, matplotlib, scipy. No other frameworks.
7. At the end of each phase, stop for human review. Do not start the next phase on your own.
8. When unsure, write the question to `QUESTIONS.md` and stop rather than guess.
9. Report numbers as measured. Do not interpret results beyond the decision table in `PLAN.md`. A negative result is a valid result.
10. Do not commit datasets, tokenized data or checkpoints. Keep them under `data/cache/` and `checkpoints/`, both git-ignored. Do commit `results/` (JSON, CSV, plots).
11. Gates in Phase 1 check that the models are properly trained. If one fails, the fix is a change to `PLAN.md` made by the human reviewer, never by you. Run only the recipes `PLAN.md` declares.

## Hardware

- Mac Studio, Apple Silicon, 64 GB unified memory. Use `torch.device("mps")` and set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Train in float32. In Task 0 also try float16 autocast for 50 steps; use it only if it runs without errors, loss matches float32 within 1%, and it is faster.
- Run SVD and eigenvalue steps on CPU in float64 (numpy or torch CPU).
- Start long runs under `caffeinate -i`. While a run trains, wait; check the log at most every 5 minutes.
- Use a Python virtual environment in `.venv/` and pin versions in `requirements.txt`.
