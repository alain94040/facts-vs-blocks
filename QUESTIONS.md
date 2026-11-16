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
