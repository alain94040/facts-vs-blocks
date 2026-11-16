# Questions for human review

## Task 0: float16 autocast

50-step trial on random tokens, seed 0 (`results/task0/throughput.json`):

| | tokens/s | loss step 1 | loss step 50 |
| --- | --- | --- | --- |
| float32 | 142.5k | 8.368 | 8.320 |
| fp16 autocast (no GradScaler) | 213.7k | 8.368 | 8.377 |

fp16 meets the literal criteria (no errors, last-10 mean within 0.69% of float32, faster), but its
loss did not fall and gradient norms were 2-10x larger than float32. I chose **float32**.
Question: keep float32 for Task 2, or investigate fp16 (e.g. GradScaler / fp32 attention)?

## Task 1: choices made where the plan was silent (please confirm at Gate 1)

1. **Names, cities, universities, companies are all invented** syllable words (fixed seed), unique
   across every pool, so no person name collides with a TinyStories name. Some look odd
   (e.g. "Quurcaquys"). Majors are 100 real subjects.
2. **Full name in every sentence**, attribute always at the end of its sentence, so a recall prompt
   is the sentence prefix (`recall_prompt` in `data/make_bios.py`).
3. **Set sizes:** A = 32,000 and B = 32,000, disjoint full names (first/last name lists are
   shared). I plan to take main-2k ⊂ main-8k ⊂ main-32k as the first N of set A, fork-A = first
   8,000 of A (same people as main-8k), and fork-B = first 8,000 of B.
4. **Tokenizer** trained on all 2.12M train stories + 100,000 biographies (training templates
   only, people from A and B). Template 5 text was not shown to the tokenizer.
5. **Data volume for Task 2:** a biography averages 117.3 tokens. 32,000 people × 100 appearances
   ≈ 375M biography tokens for main-32k, more than its 300M story tokens (main-8k ≈ 94M,
   main-2k ≈ 23M). So runs differ a lot in total tokens/steps. Is this intended, or should the
   total token budget be held fixed?

## Answers from Gate 1 review (2026-11-16)

- **fp16:** keep float32. Good call; closed. Do not spend time on fp16.
- **Task 1 choices 1 to 4:** all confirmed. Nested subsets (2k in 8k in 32k) and fork-A sharing
  people with main-8k are what we want. Shared first-name and last-name pools are good: the model
  must use the full name.
- **Choice 5, data volume:** exposures stay fixed at 100 per person and stories fixed at 300M, so
  run lengths differ. A control run `ctrl-8k-x4` is added to separate the number of facts from the
  amount of biography text. See the updated Task 2 in `PLAN.md`.
- **Gate 1:** passed. Proceed to Task 2 as rewritten in `PLAN.md`.

## Task 2a: questions asked and answered before any pilot ran (2026-11-16)

1. **Pool words that are also story words.** 4 of the 1,550 pool root words occur as real words
   in TinyStories train: Toto, Sora, Mater, Damon (146 stories; none in validation). Adding them as
   tokens would change those stories' tokenization and break "story token counts unchanged".
   **Answer: keep the Task 1 story tokens.** Story streams are bit-identical to Task 1 (asserted);
   biographies use the new tokenizer. Recorded in `results/task2a/prepare/metrics.json`.
2. **Name-copy position.** With `add_tokens` a name is `[First] [bare space] [Last]`, so the token
   right after the first name is always the space. **Answer: score the prediction at the bare-space
   position** (it must be the last-name token). The first-name position predicting the space is
   reported separately as `first_name_predicts_space`, not gated.

## Answers from Task 2a review (2026-11-16)

- Both Task 2a answers above are confirmed by the reviewer.
- Pilots: see `docs/LOG.md`. Proceed with Task 2b in `PLAN.md` (P3, then the full batch if it passes).
