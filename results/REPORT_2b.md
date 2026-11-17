# Facts vs building blocks: Phase 2b report

Numbers as measured. No interpretation beyond marking which rows of the `PLAN.md` Phase 2b table
apply. Every script was developed on the pilots first and then run once on the real models:

- `P2-main-2k-s0`, plus `P1-main-2k-s0` where two models were needed.
- Pilot outputs are in `results/task3b_pilot`, `task4b_pilot` and `task5b_pilot`. Task 3 was also
  run on P1, in `task3_pilot`, so the pilot probe-transfer step had a second model.

No method was changed after a real result, and no new question came up.

Commits: Task 3b `195a037`, Task 4b `5d64ed1`, Task 5b `20792fe`.

## Task 3b: where the per-person content is stored

Script: `analysis/task3b.py`. Per-run results are in `results/task3b/<run>/metrics.json`, the
summary is in `results/task3b/summary.csv`, and the plots are `K_L.png`, `PR_hist.png` and
`probes.png`.

Two checks on the pilot passed:

- Per-person attribution matched `I_D` computed on one person's 24 prompts alone, to within 5e-9.
- Probes reach training accuracy 1.00 from layer point 1 onward.

**1. Layer-matched random baseline.** Each cell gives recall / STORY loss / FORMAT loss /
name-copy. The random column is the mean of 3 draws with the same neurons per layer as the
knowledge set.

| Run | K | Unablated | Knowledge set | Layer-matched random |
| --- | --- | --- | --- | --- |
| main-2k s0 | 256 | 0.960 / 1.694 / 0.414 / 0.897 | 0.246 / 1.862 / 0.929 / 0.271 | 0.734 / 1.790 / 0.451 / 0.853 |
| main-2k s1 | 256 | 0.951 / 1.696 / 0.425 / 0.853 | 0.284 / 1.854 / 1.042 / 0.207 | 0.748 / 1.785 / 0.455 / 0.846 |
| main-8k s0 | 128 | 0.972 / 1.698 / 0.300 / 1.000 | 0.443 / 1.838 / 0.488 / 0.995 | 0.764 / 1.743 / 0.310 / 0.999 |
| main-8k s1 | 256 | 0.970 / 1.699 / 0.300 / 1.000 | 0.246 / 1.871 / 0.478 / 0.943 | 0.564 / 1.800 / 0.317 / 0.997 |
| main-32k s0 | 64 | 0.685 / 1.748 / 0.301 / 0.999 | 0.283 / 2.000 / 0.867 / 0.782 | 0.641 / 1.766 / 0.301 / 0.999 |
| main-32k s1 | 128 | 0.774 / 1.749 / 0.297 / 1.000 | 0.325 / 1.922 / 1.112 / 0.333 | 0.574 / 1.791 / 0.305 / 1.000 |
| ctrl-8k-x4 s0 | 256 | 0.998 / 1.695 / 0.305 / 1.000 | 0.297 / 2.035 / 0.896 / 0.341 | 0.756 / 1.791 / 0.360 / 0.994 |
| fork-A s0 | 8 | 0.987 / 1.703 / 0.300 / 1.000 | 0.482 / 1.772 / 0.468 / 0.995 | 0.975 / 1.708 / 0.310 / 1.000 |
| fork-B s0 | 8 | 0.984 / 1.702 / 0.299 / 1.000 | 0.445 / 1.765 / 0.509 / 0.968 | 0.971 / 1.706 / 0.307 / 1.000 |

**2. Per-layer K_L.** The smallest k (8 to 1024, top by `I_FACT` within one layer) at which recall
halves. "none" means it never halves.

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| main-2k s0 | 128 | none | none | none | none | 1024 |
| main-2k s1 | 256 | none | none | none | none | 1024 |
| main-8k s0 | 128 | none | 1024 | 1024 | 1024 | 1024 |
| main-8k s1 | 256 | 1024 | 1024 | 1024 | 1024 | 1024 |
| main-32k s0 | 128 | 512 | 1024 | 1024 | 1024 | 1024 |
| main-32k s1 | 128 | 512 | 512 | 512 | 512 | 1024 |
| ctrl-8k-x4 s0 | 256 | none | 1024 | 1024 | 1024 | 1024 |
| fork-A s0 | 8 | 1024 | 1024 | 256 | 512 | 512 |
| fork-B s0 | 8 | 1024 | 1024 | 512 | 512 | 512 |

**3. Participation ratio** over people 0 to 199. Kept neurons are the top 20% of `sum_p I_p`
(1,229 per run). Each cell gives the median PR, then in brackets the count of kept neurons with
PR below 10 and the number kept in that layer.

| Run | L0 | L1 | L2 | L3 | L4 | L5 |
| --- | --- | --- | --- | --- | --- | --- |
| main-2k s0 | 85.5 (0/621) | 81.1 (0/113) | 79.2 (0/132) | 77.8 (0/146) | 72.0 (0/140) | 66.7 (0/77) |
| main-2k s1 | 94.1 (0/639) | 91.7 (0/165) | 84.3 (0/93) | 87.9 (0/87) | 79.4 (0/153) | 70.7 (0/92) |
| main-8k s0 | 72.7 (0/721) | 65.9 (0/60) | 54.7 (0/142) | 48.2 (1/154) | 50.2 (1/92) | 43.3 (0/60) |
| main-8k s1 | 75.7 (0/729) | 72.2 (0/92) | 56.6 (0/93) | 49.1 (1/134) | 46.2 (0/100) | 43.6 (0/81) |
| main-32k s0 | 90.9 (0/726) | 78.9 (0/62) | 78.1 (0/123) | 66.5 (0/149) | 61.0 (0/118) | 103.9 (0/51) |
| main-32k s1 | 86.8 (0/730) | 85.5 (0/73) | 72.8 (0/105) | 48.9 (0/153) | 52.3 (0/113) | 72.1 (0/55) |
| ctrl-8k-x4 s0 | 39.0 (9/674) | 27.3 (14/72) | 27.9 (19/120) | 25.5 (17/170) | 24.6 (9/115) | 20.7 (14/78) |
| fork-A s0 | 71.9 (0/573) | 63.4 (0/57) | 56.8 (0/67) | 46.0 (3/238) | 43.7 (2/223) | 39.1 (1/71) |
| fork-B s0 | 72.4 (0/577) | 56.8 (0/49) | 56.3 (0/75) | 45.2 (0/240) | 45.7 (2/195) | 42.3 (0/93) |

Total kept neurons with PR below 10, seeds 0 / 1:

- main-2k: 0 / 0
- main-8k: 2 / 1
- main-32k: 0 / 0

**4. Own-person ablation.** People 0 to 49. Each cell gives the mean own-recall drop / mean
other-recall drop, then the ratio. Others are 49 people per person, drawn from people 50 to 249.

| Run | Top 32, all layers | Top 32, layers 1 to 5 |
| --- | --- | --- |
| main-2k s0 | 0.165 / 0.077 (2.1) | 0.154 / 0.039 (3.9) |
| main-2k s1 | 0.118 / 0.069 (1.7) | 0.131 / 0.049 (2.7) |
| main-8k s0 | 0.245 / 0.098 (2.5) | 0.193 / 0.027 (7.1) |
| main-8k s1 | 0.211 / 0.085 (2.5) | 0.199 / 0.035 (5.7) |
| main-32k s0 | 0.203 / 0.103 (2.0) | 0.107 / 0.032 (3.3) |
| main-32k s1 | 0.363 / 0.172 (2.1) | 0.161 / 0.026 (6.2) |
| ctrl-8k-x4 s0 | 0.297 / 0.061 (4.9) | 0.217 / 0.010 (21.7) |
| fork-A s0 | 0.453 / 0.274 (1.7) | 0.300 / 0.029 (10.3) |
| fork-B s0 | 0.433 / 0.245 (1.8) | 0.269 / 0.034 (7.9) |

**5. Linear read-out probes.** Best test accuracy (people 800 to 999) over the 7 layer points at
END; NAME is no higher than 0.03 in any run. Chance is 1/200 for birth city and employer city,
1/300 for university, 1/100 for major and 1/250 for employer.

| Run | Birth city | University | Major | Employer | Employer city |
| --- | --- | --- | --- | --- | --- |
| main-2k s0 / s1 | 0.010 / 0.015 | 0.025 / 0.005 | 0.295 / 0.295 | 0.020 / 0.005 | 0.015 / 0.010 |
| main-8k s0 / s1 | 0.010 / 0.010 | 0.010 / 0.005 | 0.265 / 0.285 | 0.015 / 0.015 | 0.010 / 0.015 |
| main-32k s0 / s1 | 0.010 / 0.010 | 0.010 / 0.010 | 0.025 / 0.065 | 0.010 / 0.015 | 0.010 / 0.005 |
| ctrl-8k-x4 s0 | 0.010 | 0.015 | 0.520 | 0.015 | 0.010 |
| fork-A s0 | 0.010 | 0.010 | 0.320 | 0.005 | 0.010 |
| fork-B s0 | 0.020 | 0.020 | 0.320 | 0.000 | 0.015 |
| main-0 s0 (control) | 0.010 | 0.010 | 0.015 | 0.010 | 0.005 |

**6. Probe transfer** (`results/task3b/probe_transfer.json`). Best END accuracy over layer
points:

| Direction | Birth city | University | Major | Employer | Employer city |
| --- | --- | --- | --- | --- | --- |
| fork-A probes on fork-B people 800–999 | 0.010 | 0.010 | 0.195 | 0.015 | 0.010 |
| fork-B probes on fork-A people 800–999 | 0.005 | 0.010 | 0.265 | 0.010 | 0.015 |

**7. Fork knowledge set (8 layer-0 neurons).** Mean post-GELU activation on each token type.
Token counts: 16,000 entity tokens, 62,047 other FORMAT tokens, 512,000 STORY tokens.

| | Entity tokens | Other FORMAT tokens | STORY tokens |
| --- | --- | --- | --- |
| fork-A knowledge set | 0.689 | 0.465 | 0.434 |
| fork-A, 8 random layer-0 neurons | 0.016 | −0.082 | 0.010 |
| fork-B knowledge set | 0.786 | 0.513 | 0.294 |
| fork-B, 8 random layer-0 neurons | 0.003 | −0.083 | −0.003 |

## Task 4b: where new facts can be written

**Setup:**

- **Scripts:** `data/build_ft_stream.py`, `train/finetune.py` (hyperparameters in
  `configs/finetune.json`), `train/run_task4b.sh` and `analysis/task4b.py`.
- **Results:** `results/task4b/<variant>/metrics.json` and `results/task4b/summary.csv`, with the
  plot in `results/task4b/variants.png`.
- **Stream `ft-B1k`:** 300,000 biographies of set-B people 0 to 999 (23.89M tokens) plus 105,924
  story documents (23.89M tokens), 726 steps.
- **50-step check on fork-A (F-all):** loss 1.503 → 1.239, 86k tokens/s (another job was sharing
  the GPU). Each variant took 4 to 9 minutes.

"New" means set-B people 0 to 999 and "old" means set-A people 0 to 999. "none" is fork-A before
fine-tuning.

| Variant | Trainable params | New: train templates | New: t5 seen / held out | Old: train templates | STORY loss | FORMAT loss | Name-copy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 0 | 0.005 | 0.005 / 0.004 | 0.987 | 1.7028 | 0.300 | 1.000 |
| F-all | 6,249,984 | 0.999 | 0.999 / 0.999 | 0.081 | 1.6936 | 0.418 | 0.922 |
| F-attn | 1,579,008 | 0.931 | 0.921 / 0.898 | 0.022 | 1.7073 | 0.302 | 1.000 |
| F-mlp | 3,153,408 | 0.994 | 0.992 / 0.991 | 0.116 | 1.6962 | 0.301 | 1.000 |
| F-mlp34 | 1,051,136 | 0.987 | 0.981 / 0.983 | 0.163 | 1.7037 | 0.301 | 1.000 |
| F-L0 | 788,736 | 0.023 | 0.027 / 0.019 | 0.015 | 1.7033 | 0.298 | 1.000 |
| F-emb | 1,445,376 | 0.791 | 0.771 / 0.766 | 0.132 | 1.6983 | 0.516 | 0.894 |

**Prediction check** (`results/task4b/metrics.json`):

| Variant | New recall ≥ 0.8 of F-all | Old recall within 0.05 of fork-A | STORY rise < 0.02 | Meets prediction |
| --- | --- | --- | --- | --- |
| F-mlp34 | yes (0.987 of F-all) | no (−0.824) | yes (+0.001) | **no** |
| F-mlp | yes (0.994 of F-all) | no (−0.871) | yes (−0.007) | **no** |

F-emb reaches 0.792 of F-all's new-people recall.

## Task 5b: a fair test for recurring heads

Script: `analysis/task5b.py`, on `results/task5/heads.csv` unchanged. Results are in
`results/task5b/metrics.json`, with plots `recurrence.png` and `silhouette.png`.

**Recurrence under the Gaussian null, and one-to-one matching** (`linear_sum_assignment`). The
null columns give the median and 5th percentile over 100 draws. Chance for the same-layer
fraction is 0.167.

| Pair | Features | Recur, Task 5 null | Recur, Gaussian null | Median matched distance | Null median | Null 5th percentile | Matched pairs in same layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| main-0 s0 vs s1 | STORY | 1.000 | 0.521 | 2.37 | 3.40 | 3.11 | 0.375 |
| main-2k s0 vs s1 | FULL | 1.000 | 0.667 | 2.93 | 3.97 | 3.54 | 0.500 |
| main-8k s0 vs s1 | FULL | 0.979 | 0.458 | 2.83 | 3.72 | 3.47 | 0.479 |
| main-32k s0 vs s1 | FULL | 1.000 | 0.542 | 3.06 | 4.04 | 3.80 | 0.479 |
| main-2k s0 vs main-32k s0 | FULL | 1.000 | 0.250 | 5.48 | 5.77 | 5.09 | 0.396 |
| main-8k s0 vs main-0 s0 | STORY | - | 0.250 | 3.32 | 3.57 | 3.16 | 0.417 |
| main-8k s0 vs untrained (sanity) | STORY | - | 0.000 | 15.24 | 15.34 | 15.10 | 0.188 |

**Head types.** k-means on the STORY features of the 528 trained heads, compared with 528 heads
drawn from one Gaussian fitted to them.

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 13 | 15 | 20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trained silhouette | 0.276 | 0.273 | 0.272 | 0.271 | 0.275 | 0.238 | 0.230 | 0.219 | 0.236 | 0.211 | 0.206 |
| Null silhouette | 0.249 | 0.238 | 0.226 | 0.204 | 0.208 | 0.196 | 0.183 | 0.166 | 0.147 | 0.144 | 0.139 |

- **Best k:** 2 for the trained heads (0.276) and for the null (0.249).
- **Gap (trained minus null):** +0.027 to +0.089 across k = 2 to 20; the largest is at k = 13.

Clusters at the trained best k = 2:

| Cluster | Size | Heads per layer (L0 … L5) | Story induction | Story duplicate | Previous-token | Sink | Models contributing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 204 | 18, 13, 29, 43, 43, 58 | 0.066 | 0.038 | 0.085 | 0.021 | 11 |
| 1 | 324 | 70, 75, 59, 45, 45, 30 | 0.040 | 0.027 | 0.127 | 0.013 | 11 |

## Decision table

| Result | Applies? | Basis |
| --- | --- | --- |
| Own-person ablation (layers 1 to 5) drops own recall ≥ 3× other people's, and the count of kept neurons with PR < 10 grows with N → per-person content sits in identifiable neurons outside layer 0 | **No** | Ratio 2.7 to 21.7 (8 of 9 runs ≥ 3; main-2k s1 is 2.7). Count of PR < 10: 0/0 (2k), 2/1 (8k), 0/0 (32k), which does not grow with N |
| Own and other drops within a factor of 1.5, and median PR above 50 in every layer → per-person content spread thinly | **No** | Ratios are 1.7 to 4.9 (all layers) and 2.7 to 21.7 (layers 1 to 5); median PR falls below 50 in some layer of main-8k (both seeds), main-32k s1, ctrl-8k-x4 and both forks |
| Probe test accuracy above 0.5 for at least 3 attributes at some END layer point, and main-0 near chance → shared linear read-out | **No** | Only major in ctrl-8k-x4 is above 0.5 (0.520). main-0 is at most 0.015 |
| Fork-A probes score above 0.5 on fork-B people → read-out shared between forks | **No** | Best is 0.195 (major) |
| F-mlp34 or F-mlp meets the Task 4b prediction → new facts can be confined to the MLPs | **No** | Both pass new recall and STORY loss but fail retention: set-A recall falls by 0.82 / 0.87 |
| F-emb reaches at least 50% of F-all's new-people recall → identity can live in name embeddings | **Yes** | 0.791 against 0.999 (79%) |
| Under the Gaussian null, recurrence for trained pairs is above 0.5 and below 0.2 for the untrained sanity pair → heads recur | **No** | Sanity pair 0.000. Trained pairs: 0.521, 0.667, 0.458, 0.542 (seed pairs); 0.250 (2k vs 32k); 0.250 (8k vs main-0). Not all above 0.5 |
| Trained silhouette beats the null's at its best k by at least 0.1 → small number of head types | **No** | At k = 2: 0.276 against 0.249 (+0.027) |
| Trained silhouette within 0.05 of the null at every k → no evidence of head types | **No** | The gap exceeds 0.05 at 14 of 19 values of k (largest 0.089) |

**End of Phase 2b. Stopped for human review.**
