# Experiment E: what the middle layers compute

An independent experiment on `Qwen/Qwen2.5-7B-Instruct`. It records activations and runs edited
forward passes; it trains nothing except small linear fits.

**Why.** Experiment C showed that layers 6 to 17 form a family that can stand in for each other,
and that their weights share directions but not neurons. Weights show what a layer could do.
This experiment looks at what each layer actually does to real activations, in two ways:

- **Part 1, operators.** Fit, for every layer, the best linear map from its input to the change
  it adds, then ask whether one layer's map explains another layer's change. If the family is
  one operation in several noisy copies, the maps will agree even where the weights look
  different.
- **Part 2, two-hop questions.** Follow a two-step question ("the official language of the
  country whose capital is Lima") through the layers. Find where each step is resolved, whether
  the second step reuses the first step's machinery at a later depth, and whether copying a
  later state back into an earlier layer ("back-patching", Biran et al. 2024, "Hopping Too Late")
  fixes failures. That is a targeted form of re-running the middle.

The rules in `CLAUDE.md` apply. In addition:

- All code lives in `middle/`, all outputs in `results/middle/`. Import `repeat/flow.py`,
  `repeat/data.py` and `repeat/evaluate.py`; do not edit them.
- Libraries: the same exception as Experiment C (`transformers`, `safetensors`,
  `huggingface_hub`, pinned in `repeat/requirements.txt`).
- Model in float16 on MPS; accumulated statistics in float64 on CPU.
- Develop everything on `Qwen/Qwen2.5-0.5B-Instruct` first. Its numbers are not results.
- Commit messages: `middle task N: <what>`.
- Zones as in Experiment C: ENTRY 0–5, FAMILY 6–17, OUTPUT 18–27.
- Outlier coordinates: the 16 found in Repeat task 4b. Every Part 1 number is reported twice,
  with all coordinates and with those 16 removed from the target.

## Part 1: layer operators

### Middle task 1: statistics

**Data.** 480 chunks of 256 tokens from the **train** split of `wikitext-103-raw-v1` (not the
validation split used by Experiment C), seed 0. Chunks 0–379 are FIT, chunks 380–479 are TEST.
Use positions 8 to 255 of each chunk only (the first positions behave differently because of the
attention sink). That gives about 94k FIT and 25k TEST positions.

**Per layer l**, with `x` the residual stream entering the layer and `y` the one leaving it:

- input `z = x / rms(x)`, the same scaling the layer's own RMSNorm applies, without its gain,
  with a constant 1 appended (dimension d + 1 = 3,585);
- target `u = y - x`, the layer's update (dimension d = 3,584).

Accumulate, separately for FIT and TEST, the sums `Szz = Σ z zᵀ`, `Szu = Σ z uᵀ`,
`Suu = Σ u uᵀ` and the count. Never store per-token activations. Add each batch's float32
products to float64 accumulators on CPU. Save the accumulators to `data/cache/middle/`. Memory:
about 300 MB per layer per split, so write each layer to disk as it finishes a pass if needed.

Every quantity below is computed exactly from these sums. For a map `M` (shape (d+1) by d):
`err(M) = tr(Suu) - 2 tr(Mᵀ Szu) + tr(Mᵀ Szz M)`, and `R² = 1 - err(M) / tot`, where `tot` is
`Suu`'s trace after removing the mean of `u`.

### Middle task 2: one operator per layer

For each layer, fit ridge regression on FIT: `M_l = (Szz + λ I)⁻¹ Szu`, with
`λ = c · tr(Szz) / (d+1)` and c chosen per layer from {1e-4, 1e-3, 1e-2, 1e-1} by TEST R².
(The fit's own penalty only; no evaluation score is tuned.) Report per layer:

- `R²_ll`, the TEST R² of the layer's own map: how much of what the layer does is linear in its
  input;
- the noise ceiling: fit on FIT chunks 0–189 and on 190–379 separately, and report the R² of one
  half's map on the other half's sums.

### Middle task 3: does one layer's operation explain another's

For every pair (i, j) of the 28 layers:

- `R²_ij`: TEST R² of layer j's map on layer i's sums (layer j's operation applied to layer i's
  inputs, compared with layer i's actual update);
- `R²_ij,scaled`: the same after the best single scalar `a` multiplies `M_j` (computed in closed
  form from the sums), since two layers can apply one operation at different strengths;
- the ratio `R²_ij,scaled / R²_ii`: the share of layer i's linear behaviour that layer j's
  operation reproduces.

**Null: rotated operator.** Replace `M_j` by `Q M_j Qᵀ` restricted to the residual coordinates,
with Q a random orthogonal matrix (seed 0; the constant row is left as is). This keeps everything
about `M_j` except which directions it uses. Report the same three numbers for the null.

**Read and write subspaces.** From the SVD of each `M_l` (without the constant row), take the top
64 right singular vectors (what the layer reads) and the top 64 left singular vectors (what it
writes). For every pair, report the mean cosine of the principal angles between the read
subspaces, and between the write subspaces. Null: the same with one of the two rotated as above.

Output: `results/middle/task3/pairs.csv`, heat maps (28 by 28) of the ratio, of read overlap and of
write overlap, each next to its null.

## Part 2: two-hop questions

### Middle task 4: data

Write `middle/countries.py`, a table of countries with capital, continent and official language.
Keep only countries where all three are unambiguous: one capital, one official or overwhelmingly
dominant language, no disputed status. Aim for about 120 rows, and record in the file the rule
used to drop each excluded case. Templates, each with three fixed worked examples in front,
built from rows not used as items:

| Kind | Prompt | Answer |
| --- | --- | --- |
| HOP1 | `The country whose capital is Lima is` | ` Peru` |
| HOP2-language | `The official language of Peru is` | ` Spanish` |
| HOP2-continent | `Peru is in the continent of` | ` South America` |
| TWO-language | `The official language of the country whose capital is Lima is` | ` Spanish` |
| TWO-continent | `The country whose capital is Lima is in the continent of` | ` South America` |

The **bridge** is the country. A two-hop item is **eligible** when the model gets its HOP1 and the
matching HOP2 right by greedy decoding of the answer's first token. Report accuracy for every
kind, and for TWO among eligible items.

### Middle task 5: where each step happens

For every eligible TWO item and its matching HOP1 and HOP2 items, at the last prompt token, apply
the model's final norm and unembedding to the residual stream after every layer (logit lens).
Record, per layer, the rank of the bridge's first token and of the answer's first token. Define:

- `L_bridge`: the first layer after which the bridge is in the top 10 in the TWO prompt (or none);
- `L_answer2`: the same for the answer in the TWO prompt;
- `L_answer1`: the same for the answer in the HOP2 prompt (one step, bridge given in the text).

Report their distributions, and `delay = L_answer2 - L_answer1` for items where both exist.

**Reuse test.** For each item, sum the updates of the three layers ending at `L_answer1` in the
HOP2 prompt, and of the three layers ending at `L_answer2` in the TWO prompt, at the last token.
Report the cosine between the two sums (outlier coordinates removed). Null: the same cosine
between an item's TWO sum and the HOP2 sum of a different item with a different answer (one fixed
random pairing, seed 0). If the second step reuses the one-step machinery later, real cosines
clearly exceed the null.

### Middle task 6: back-patching

For each TWO item (eligible ones, failed and solved, reported separately), and for target layer
i and source layer j with j > i, both even and in 2..26: take the residual stream after layer j at
the last prompt token, write it in place of the residual stream after layer i at that token, and
run layers i+1 to 27 again. Success: the answer's first token is the greedy top prediction.

Report, per (i, j): the fraction of failed items fixed and the fraction of solved items broken.

**Control.** The same patch with the source state taken from a different item (same layer j,
fixed random pairing, seed 0). A real effect fixes clearly more than this control.

Output `results/middle/task6/backpatch.csv` and two heat maps (fixed, broken).

### Middle task 7: report

Write `results/middle/REPORT.md` with the tables and plots, and mark which rows apply:

| Result | What it means |
| --- | --- |
| Family layers have own-map R² of at least 0.5 | The family's updates are mostly linear in their input; operator comparisons are meaningful |
| Within the family, mean `R²_ij,scaled / R²_ii` is at least 0.5 for pairs up to 4 apart, and the rotated null is at most 0.1 | Neighbouring family layers apply largely the same operation |
| The ratio stays at least 0.3 across the whole family | The family is one operation repeated, at different strengths |
| The ratio inside the family is clearly above OUTPUT's (by at least 0.2) | Operation sharing is special to the family, matching the swap map |
| Read overlap high but write overlap low (or the reverse) within the family | Layers share what they look at but not what they write (or the reverse) |
| Median delay is at least 2 layers and `L_answer2` often lands in OUTPUT | The second step happens late, as Biran et al. found |
| Reuse cosines clearly exceed the null | The second step reuses the one-step machinery at a later depth: one block, two uses |
| Back-patching fixes at least 20% of failures for some (i, j) with i in the FAMILY, and well above the control | Re-running the middle on an unfinished state helps, a targeted form of RYS |
| Back-patching fixes no more than the control | Re-running the middle does not rescue the second step here |

Stop for review.

## Budget

Task 1: one pass of about 123k tokens with accumulation, about 15 minutes. Tasks 2 and 3: linear
algebra on 3,585 by 3,585 matrices, about 30 minutes. Tasks 4 and 5: under 10 minutes. Task 6:
about 90 (i, j) pairs, each a partial forward pass over at most a few hundred short prompts,
about 20 minutes. Total about 1.5 hours.

## Limits

- A linear map is only the first-order part of a layer. `R²_ll` says how much that misses.
- One text distribution (wikitext). Layers may apply different operations on code or arithmetic.
- The logit lens reads intermediate layers imperfectly, so `L_bridge` and `L_answer` are
  estimates; they are used for timing, not as proof that a step happened.
- Countries are well-known facts; the second hop is easy for the model once the bridge is known,
  which is the condition the method needs.

## Follow-up (added after the review of the report)

### Middle task 8: reuse null with the same answer

Task 5's reuse null pairs items with **different** answers, while the real pair always shares the
answer. A high real cosine can therefore mean only that both sums write the same answer token,
not that the same machinery is reused. Add a second null:

- For each eligible item, pair it with a different item that has the **same** answer (same
  language or same continent). Use a seeded random cycle as in the fixed task 5 null: each item
  takes the next item after it with the same answer, and no partner is used more than 3 times.
  Items whose answer is unique are excluded; report how many.
- Cosine between the item's TWO sum (ending at its `L_answer2`) and the partner's HOP2 sum (ending
  at the partner's `L_answer1`), outlier coordinates removed, exactly as in task 5.
- Report per kind: real, same-answer null, different-answer null (task 5), and real minus
  same-answer null with a bootstrap 95% CI. All three are computed on the same items, so restrict
  real and the task 5 null to the items that have a same-answer partner.

Add one row to the report's decision table:

| Result | What it means |
| --- | --- |
| Real minus same-answer null is at least 0.1 with a CI above 0, for both kinds | The second step reuses item-specific one-step machinery, beyond writing the same answer |
| Real minus same-answer null is below 0.05 | The reuse cosine reflects the shared answer only |

Output `results/middle/task8/metrics.json` and append a section to `results/middle/REPORT.md`.
Minutes of compute. Commit as `middle task 8: <what>`.
