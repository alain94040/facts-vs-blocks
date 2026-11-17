# Experiment D: do the middle layers share one set of directions?

An independent experiment on the same Qwen models as Experiment C. It reads and edits weights;
it trains nothing.

**Why.** Experiment C found a "family" of layers in Qwen2.5-7B, roughly layers 6 to 17. They can
stand in for each other, and the more alike two layers' weights are, the cheaper the swap. But
neurons have no one-to-one partners across layers, so whatever they share is a set of
directions in the residual stream, not individual units. This experiment measures that shared
set directly. If one basis, much smaller than a layer, explains most of every family layer,
that basis is the building block, and the compression it allows can be read off.

The rules in `CLAUDE.md` apply. In addition:

- All code lives in `basis/`, all outputs in `results/basis/`. Reuse `repeat/flow.py`,
  `repeat/data.py` and `repeat/evaluate.py` by importing them; do not edit them.
- Libraries: the same exception as Experiment C (`transformers`, `safetensors`,
  `huggingface_hub`, pinned in `repeat/requirements.txt`).
- Linear algebra in float64 on CPU. Never hold more than two layers' weights in memory.
- Commit messages: `basis task N: <what>`.
- Develop everything on `Qwen/Qwen2.5-0.5B-Instruct` first. Its numbers are not results.

## 1. Definitions

**Zones** (fractions of depth so they carry over to 14B): ENTRY = layers below 21% of depth,
FAMILY = 21% to 64%, OUTPUT = above 64%. On 7B that is ENTRY 0–5, FAMILY 6–17, OUTPUT 18–27. On
14B (48 layers): ENTRY 0–9, FAMILY 10–30, OUTPUT 31–47. Write the exact lists to
`results/basis/zones.json` in task 0.

**Matrices.** For each layer, after folding RMSNorm gains into the matrices that read from the
residual stream (as in Repeat task 4b): `q`, `k`, `v`, `gate`, `up` read from the residual stream;
`o` and `down` write to it. Each has one **residual side** of dimension d (3,584 on 7B, 5,120 on
14B, 896 on 0.5B): the columns of the reading matrices, the rows of the writing ones. Write every
matrix as `W` with shape (d, n): residual side first, its other side second.

The residual side is the only side whose coordinates mean the same thing in every layer, so all
sharing is measured there.

**Own basis.** For one layer's matrix `W`, the top k left singular vectors of `W`. Rank k
cannot keep more of `W` than this.

**Shared basis.** For a zone Z and a matrix type, `S = sum over layers l in Z of W_l W_l^T`
(d by d). Its top k eigenvectors `U_k` are the directions that the layers of the zone use most,
together.

**Kept energy.** For a basis `U` and a matrix `W`: `|U U^T W|_F^2 / |W|_F^2`.

**Sharing efficiency** at k: kept energy with the zone's shared basis divided by kept energy
with the layer's own basis, averaged over the zone's layers. 1 means the shared basis is as good
as each layer's private one.

**Null: rotated layers.** For each layer in the zone, apply an independent random orthogonal
matrix `Q_l` (d by d, seed = layer index) to the residual side: `W_l → Q_l W_l`. This keeps every
layer's singular values and own-basis energy exactly, and destroys any directions the layers
share. Recompute the shared basis and the sharing efficiency on the rotated layers.

**Outlier coordinates.** On 7B, use the 16 outlier coordinates found in Repeat task 4b. On 14B, find them the same way (Repeat task 4b step 1) in task 3. Every energy
measurement is reported twice: on all coordinates, and with those 16 residual rows set to zero.

## 2. Tasks

### Basis task 0: setup

Load the development model, compute everything in task 1 for its family, and check:

- with k = d, every kept energy is 1 (to 1e-9);
- own-basis energy is at least shared-basis energy for every layer and k;
- rotated null: own-basis energy is unchanged by the rotation (to 1e-9).

Then the same checks on 7B for one layer. Stop for review if any fails.

### Basis task 1: how much is shared (7B, weights only)

For each zone (ENTRY, FAMILY, OUTPUT), each of the 7 matrix types, and k = d/16, d/8, d/4, d/2:
kept energy with own basis, with the zone's shared basis, and with the rotated-null shared basis;
and the sharing efficiency for real and null.

Also, for the family only: the shared basis of the **other** zones applied to family layers
(does the family share directions with everyone, or only with itself?).

Output `results/basis/task1/energy.csv` and one plot per matrix type: sharing efficiency against
k, one line per zone, real solid and null dashed.

### Basis task 2: does the shared basis keep the model working (7B)

Replace weights and measure with Experiment C's evaluation on the SEARCH halves (TEXT, REASON,
RECALL-HARD). Each configuration below edits every layer of the FAMILY at once.

| Config | Each family matrix `W` becomes | Meaning |
| --- | --- | --- |
| `base` | `W` | Unmodified |
| `own(k)` | `U_own U_own^T W` (that layer's own top k) | Best possible rank k, no sharing |
| `shared(k)` | `U_k U_k^T W` (the zone's shared top k) | Rank k through one shared basis |
| `null(k)` | the same, with the rotated-null shared basis rotated back | Rank k through a basis with nothing shared |

Run each at k = d/8, d/4, d/2, for two groups separately: the MLP matrices (`gate`, `up`, `down`)
and the attention matrices (`q`, `k`, `v`, `o`). That is 3 configs × 3 values of k × 2 groups =
18 runs plus `base`.

For every run also record the **parameter count** of the family under that scheme:

- original: sum over family layers of d × n per matrix;
- `own(k)`: per layer d × k + k × n;
- `shared(k)`: d × k once per matrix type, plus per layer k × n.

Report the compression ratio of the family (original divided by scheme) next to the loss change.

Output `results/basis/task2/edits.csv` and one plot: TEXT loss change against compression ratio,
one line per config and group.

### Basis task 3: same on 14B (weights first, then a short check)

Load `Qwen/Qwen2.5-14B-Instruct`. Run task 1 in full. Then run task 2 only for `base`,
`shared(d/4)` and `own(d/4)`, both groups, on the SEARCH halves.

### Basis task 4: report

Write `results/basis/REPORT.md` with the tables and plots, and mark which rows apply:

| Result | What it means |
| --- | --- |
| Task 0 checks fail | Stop; no row below applies |
| Family sharing efficiency at k = d/4 is at least 0.9 and the null's is at most 0.7, for the MLP matrices | The family uses one common set of directions for its MLPs |
| Same for the attention matrices | The same holds for attention |
| Family efficiency clearly above OUTPUT's (by at least 0.1) at k = d/4 | Sharing is special to the family, matching Experiment C's swap map |
| Other zones' bases keep the family's energy as well as its own | The shared directions are global to the model, not a family trait |
| `shared(k)` costs at most 0.02 TEXT nats more than `own(k)` at the same k | Sharing the basis is nearly free; the per-layer part is the coefficients |
| Some `shared(k)` keeps TEXT within +0.05 of `base` at a family compression of at least 2 | The family can be compressed by basis sharing without training |
| `shared(k)` damage close to `null(k)` | The shared basis is no better than a basis with nothing shared |
| 14B rows agree with 7B rows | The structure is not specific to one model |

Stop for review.

## 3. Budget

Weights only for tasks 1 and 3a: eigen-decompositions of 3,584 by 3,584 matrices (5,120 on 14B),
a few minutes each, about 30 minutes in all. Task 2: 19 evaluations at about 14 s each plus the
time to edit weights, under 30 minutes. Task 3b: about 20 minutes. Total about 1.5 hours.

## 4. Limits

- Energy in the weights is not importance for the output. Directions with little energy can
  matter a lot. Task 2 is the check on that.
- Only the residual side is shared. The other side (neurons, head dimensions) has no common
  coordinates across layers without the alignment of Repeat task 4b.
- No training. A small fine-tune after sharing usually recovers much of the loss (Basis Sharing,
  arXiv 2410.03765; Relaxed Recursive Transformers, arXiv 2410.20672). Training-free numbers
  are a lower bound on what sharing allows.
