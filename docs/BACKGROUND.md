# Background

Context for anyone, human or agent, working in this repository. It explains why the experiment
exists and what is already known. **`PLAN.md` stays authoritative for what to do.** Nothing here
changes a task, a gate or a threshold.

## 1. The research question

Original hypothesis (John): the weights of a large language model are built from a small library
of elementary functional "building blocks", perhaps thousands. Gradient descent finds each block
many times in numerically different forms, so the repetition is hidden. If blocks could be
recognised and shared, a model might shrink by about 10X.

Stated so it can be tested:

- **H1** The weights decompose into components that fall into a small number K of functional
  equivalence classes.
- **H2** Members of a class compute the same function up to known symmetries plus small noise.
- **H3** Replacing every member with one shared prototype, plus a cheap per-instance transform,
  keeps the loss within a small tolerance.
- **H4** K is much smaller than the number of instances, so storage drops by about 10X.

Current working version (v2): a model is two things mixed together. One part is **knowledge**
(facts), which is close to random and cannot be shared. The other part is **procedures**, which
may be the reusable blocks. This repository tests whether that split exists at all in a small
model. The compression question comes later and only matters if the answer is yes.

### The arithmetic that sets the bar

If a fraction `f` of the parameters can be shared with reduction `r` and the rest stays as is:

```
C = 1 / ((1 - f) + f / r)
```

Even with unlimited `r`, 10X needs `f >= 0.9`. If half the model is knowledge, the ceiling is 2X.
So the quantity that matters is **coverage**: what share of parameters is reusable.

## 2. Corrections to common intuitions

These shaped the design. Keep them in mind when reading results.

1. **Noise is the smaller effect.** Two components that look different usually differ because
   they sit in a different basis (a symmetry) or because they really do different things, not
   because of rounding. Never compare raw weight values across components.
2. **Symmetries of a transformer.** Neurons in an MLP layer can be permuted. Within an attention
   head, queries and keys can be transformed by any invertible matrix as long as the product
   `W_Q W_K^T` is unchanged, and the same for `W_V W_O`. So compare the products, or compare
   behaviour on real activations.
3. **Superposition.** Models represent more features than they have neurons by packing them in
   non-orthogonal directions. A single neuron is often not a meaningful unit. Results at neuron
   level are approximations.
4. **Searching finds structure in noise.** If you try enough transforms or clusterings, random
   weights will show patterns too. This is why every analysis also runs on an untrained model and
   a shuffled model, and a finding only counts if it clearly beats both.
5. **Real facts are not random.** Real-world facts have regularities that models exploit. The
   synthetic biographies here are independent and uniform by construction, so a clean split in
   this repository is an upper bound on how clean it would be in a real model.
6. **Fact storage needs varied text.** A model that sees a fact in only one wording memorises the
   string without being able to answer questions about it. Varied templates and shuffled sentence
   order fix this (Allen-Zhu and Li, arXiv 2309.14316). That is why the biographies are rendered
   differently on every appearance, and why template 5 is held out.

## 3. What the literature says

| Area | Result | Relevance |
| --- | --- | --- |
| Weight sharing, codebooks | Deep Compression (1510.00149), AQLM (2401.06118), QuIP# (2402.04396): about 2 bits per weight with modest loss | Numerical redundancy is large, but this is sharing of scalars, not of mechanisms |
| Cross-layer sharing | ALBERT (1909.11942), Relaxed Recursive Transformers (2410.20672), Basis Sharing (2410.03765) | Closest to "prototype plus per-instance correction"; gains of 2X to 3X |
| Symmetries, alignment | Entezari et al. (2110.06296), Git Re-Basin (2209.04836), ZipIt (2305.03053) | Independently trained networks become nearly interchangeable after permutation alignment |
| Universality | Universal Neurons in GPT2 (2401.12181): only 1 to 5% of neurons recur across seeds. Clock and Pizza (2306.17844): same task, different algorithms | Recurring components exist but are a minority |
| Superposition | Toy Models of Superposition (Elhage et al. 2022), sparse autoencoders (Bricken et al. 2023) | The natural unit is a direction, not a neuron |
| Parameter decomposition | APD (2501.14926), SPD (2506.20790) | Direct ways to define a mechanism in weight space; only shown on small models |
| Weight-sparse training | Weight-sparse transformers have interpretable circuits (2511.13653) | A way to get disentangled circuits by training, at a cost in capacity |
| Factual memory | ROME (2202.05262): facts are recalled through mid-layer MLPs. Knowledge capacity laws (2404.05405): about 2 bits of facts per parameter | Supports a knowledge part located in MLPs, and limits compression |
| Facts outside the weights | RETRO (2112.04426), memory layers (2412.09764) | Moving facts to separate storage lets the rest be much smaller |
| Precision | Scaling Laws for Precision (2411.04330) | Longer-trained models compress worse |

Numbers in parentheses are arXiv identifiers.

**Supports the hypothesis:** hidden equivalence under symmetry is real; some mechanisms recur in
nearly every model; layers can be tied with small corrections.

**Cuts against it:** universality covers a minority of the network; facts do not repeat; no
published result reaches 10X through block reuse.

**Open:** what share of parameters is procedural versus memorised, and how the number of distinct
blocks grows with model size.

## 4. Known building blocks

Components that interpretability research has named. Task 5 scores heads against the first group.

**Attention heads**

| Block | What it does | How to detect it |
| --- | --- | --- |
| Previous-token head | Attends to the token just before | High attention on offset -1 |
| Induction head | After `A B ... A`, attends to `B` and predicts it | On a repeated random sequence, attention from a token to the one after its earlier copy |
| Duplicate-token head | Attends to earlier copies of the current token | Attention to same-token positions |
| Attention sink | Parks attention on the first position, a no-op | High attention on position 0 |
| Copying head | Raises the probability of the token it attends to | Positive eigenvalues of the OV circuit through the embeddings |
| Name mover, S-inhibition | Parts of the indirect-object circuit (2211.00593) | Task specific; not expected in this model |
| Successor head | Monday to Tuesday, 3 to 4 (2312.09230) | Task specific |
| Copy suppression | Lowers a token the model would wrongly repeat (2310.04625) | Negative copying score |

Induction needs two heads in different layers: a previous-token head early, an induction head
later. Induction heads typically appear in a sudden drop in loss early in training.

**MLP components**

| Block | What it does |
| --- | --- |
| Key-value memory | An input pattern triggers a stored output. The general mechanism behind fact recall (2012.14913) |
| Universal neurons | Track position, token frequency, letter; recur across seeds (2401.12181) |
| Prediction and suppression neurons | Late layers; push groups of next tokens up or down |
| Entropy neurons | Final layer; change confidence without changing the top token (2406.16254) |

The pattern that matters here: attention blocks are few in kind and recur. MLPs use one general
mechanism filled with unique contents. That is the split this experiment looks for.

## 5. How the experiment maps to the question

| Design choice | Why |
| --- | --- |
| Same stories, varying number of people (0, 2k, 8k, 32k) | The number of facts is the only thing meant to change, so what grows with it is the knowledge part |
| `ctrl-8k-x4` | Same amount of biography text and training as main-32k with 4x fewer facts, to separate "more facts" from "more text" |
| Random independent attributes | Nothing to generalise, so recall must come from stored memory |
| Invented names, shared first and last name pools | No overlap with story vocabulary; the model must use the full name |
| Held-out template 5 | Tests that facts are stored usably, not as memorised strings |
| Fork runs from a shared trunk | Two models in the same internal coordinates, so swapping parts needs no alignment |
| Swap control from an unrelated seed | Shows that a successful swap is not trivially easy |
| Induction check | Confirms the model learned at least one known procedure before analysing procedures |
| Untrained and shuffled controls | Guard against structure created by the analysis itself |
| Two seeds | Lets Task 5 measure which neurons recur across independent runs |

## 6. Terms

- **Nats:** loss in natural-log units. A uniform guess over 4096 tokens costs ln(4096) = 8.32.
- **Ablation:** switching a component off (here, replacing its activation with its mean) to see
  what depends on it.
- **Attribution:** an estimate of a component's effect on the loss from weight times gradient,
  without running an ablation for each one.
- **OV and QK circuits:** for one attention head, `W_Q W_K^T` decides where it looks and
  `W_V W_O` decides what it writes. Both are invariant to the head's internal basis.
- **Effective rank:** exp of the entropy of the normalised singular values; a soft count of the
  directions a matrix really uses.
- **Knowledge set:** defined in Task 3 as the smallest set of MLP neurons whose ablation halves
  fact recall while raising story loss by under 0.05 nats.
- **Coverage:** the share of parameters that belongs to reusable classes.

## 7. After this experiment

Depending on the result (see the decision table in `PLAN.md`):

- If the split is clean: count block classes inside the fixed part with symmetry-aware
  comparison, then repeat with real text where facts are structured.
- If facts and procedures are entangled: try a weight-sparse model or an architecture with an
  explicit memory layer, and test whether the split can be imposed by design.
- In both cases, measure coverage before making any compression claim.

## 8. Research directions from the literature

Ideas adjacent to this experiment. None is part of the current plan.

| Direction | What it suggests | Source |
| --- | --- | --- |
| Separate memory by design | Do not search for the split; build it. Put facts in a lookup table or retrieval store and train a small dense core beside it | Memory layers (2412.09764), RETRO (2112.04426) |
| Facts as rank-one edits | A fact can be written into an MLP as a rank-one update, so knowledge may be a sum of key-value terms that can be counted or subtracted | ROME (2202.05262), MEMIT (2210.07229) |
| Caution on localisation | Where a fact shows up under causal tracing is not where editing works best, so "where is the fact" has several answers | Hase et al. (2301.04213) |
| Memorisation is spread out | In vision models memorised examples depend on a small set of neurons scattered over layers, not on particular layers | Maini et al. (2307.09542) |
| Relations as reusable blocks | A relation such as "works in" is often one linear map shared by all subjects. The block is the relation; the entries are the data | Hernandez et al. (2308.09124) |
| Memorising versus generalising circuits | Both compete inside one model; weight decay favours whichever is cheaper per unit of loss. Gives a quantitative reason for when a model stores and when it computes | Varma et al. (2309.02390), Nanda et al. (2301.05217) |
| Shared random bases | One frozen random matrix shared by all layers plus tiny per-layer vectors can replace per-layer adapters. Sharing can work without the shared thing being learned | VeRA (2310.11454) |
| Subnetworks of random weights | Good networks exist inside untrained random ones, so selection may matter more than values | Ramanujan et al. (1911.13299) |
| Shared layers plus corrections | The most direct compression test of the block idea: one shared layer, a low-rank correction per depth | Relaxed Recursive Transformers (2410.20672) |
| Mechanisms in weight space | Decompose weights into components that each do one thing, then compare components across layers and models | APD (2501.14926), SPD (2506.20790) |
| Disentangling by training | Force most weights to zero and circuits become small and separable | Weight-sparse transformers (2511.13653) |
| Compression view | A language model is a compressor; ask how many bits each parameter carries and which bits are shared | Delétang et al. (2309.10668), Allen-Zhu and Li (2404.05405) |

Two of these connect directly to results already seen here. The pilots showed stored facts
overriding in-context copying (name-copy errors that are memorised last names), which is the
competition described by the circuit-efficiency work. And the "relation as a shared map" idea
predicts that the six attributes here are six blocks, with the people as data, which Task 3 could
be extended to test.
