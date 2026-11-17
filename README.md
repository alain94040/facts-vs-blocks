# facts-vs-blocks

Does a small language model split into a **fact store** (grows with the number of facts, not reusable)
and a set of **reusable building blocks** (fixed, shared across models)?

This repository holds a controlled experiment on tiny transformers trained on TinyStories mixed with
synthetic biographies. It is the first step of a larger question: can LLM weights be compressed by
recognising and re-using a small library of functional blocks?

- `PLAN.md` is the full task list, with a deliverable and a numeric gate per task.
- `CLAUDE.md` holds the rules for any coding agent working here. Read it first.
- `loop/PLAN.md` is a second experiment: more reasoning steps per weight by looping a block.
- `docs/BACKGROUND.md` is the research background: hypothesis, literature, known building blocks.
- `results/` is where every run writes its metrics.

Hardware target: one Mac Studio (Apple Silicon, 64 GB unified memory), PyTorch MPS backend.
