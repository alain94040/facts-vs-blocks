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
