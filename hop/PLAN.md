# Experiment H: does Gemma pass through "Mexico"? Swap test on the Mexico-Spanish graph

An independent experiment on an existing open model. It builds one attribution graph and runs
feature interventions with pretrained transcoders. It trains nothing.

**Why.** Neuronpedia hosts an attribution graph that nobody has explained yet: Gemma-2-2B, prompt
`The language most commonly spoken in the country south of the United States is`, answer
` Spanish`. Getting there takes two hops: "south of the United States" → Mexico (the **bridge**),
then Mexico → Spanish. "Mexico" never appears in the prompt. There are two possible routes:

- **Hop:** the model represents Mexico internally, then looks up Mexico's language.
- **Shortcut:** the model goes from "south of the United States" to Spanish without a
  Mexico-specific state (for example through a vague "Latin America" association).

Both can operate at once. In the published Dallas → Texas → Austin graph, the hop and a direct
shortcut both contribute. A graph alone cannot tell the routes apart, because it shows what is
correlated with the answer, and a 16k per-layer transcoder leaves a large error term. This
experiment decides the question by intervening: it **removes the Mexico features and puts another
country's features in their place**. If the answer follows the swap (Brazil → Portuguese), the
hop is real. If Spanish survives the removal, a shortcut carries it. The same test then runs on a
family of two-hop prompts, so the answer does not rest on one prompt.

It complements Experiment E. E found that the second hop on Qwen 7B shows no delay, and that the
last token carries no item-specific information until about layer 18–20. This experiment measures
where the bridge lives in a second model, with features instead of whole-layer states.

The rules in `CLAUDE.md` apply. In addition:

- All code lives in `hop/`, all outputs in `results/hop/`. Do not edit any file outside `hop/` and
  `results/hop/`. You may import `circuit/ct_compat.py` and `middle/countries.py`; do not edit them.
- Do not start any run while another experiment is using the GPU. Check with `ps` first.
- Environment: reuse `circuit/.venv` (circuit-tracer 0.5.0, TransformerLens backend, safetensors
  0.7.0) without changing it. Import `circuit/ct_compat.py` before loading a model. If a package is
  missing, write the question to `QUESTIONS.md` and stop.
- Model weights and transcoders go to `data/cache/hf/` (git-ignored). Check free disk space before
  downloading and write the sizes to `results/hop/task0/metrics.json`.
- Run the model in float32 on MPS. Experiment G found that bfloat16 shifts logits by far more than
  the checks below allow. Do statistics in float64 on CPU.
- Commit messages: `hop task N: <what>`.
- Seed 0 for every random choice.
- One review stop, at the end (Task 6), plus any gate failure.

## 1. Model and tools

| Item | Value |
| --- | --- |
| Model | `google/gemma-2-2b` (base, 26 layers, d_model 2304). The model is gated on Hugging Face: if the download is refused, ask John to accept the licence and log in, then stop. |
| Transcoders | `mntss/gemma-scope-transcoders`, the GemmaScope 16k per-layer transcoders, loaded with `ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma")`. These are the "GEMMASCOPE-TRANSCODER-16K" set named on the Neuronpedia graph. |
| Graphs | `circuit_tracer.attribute(prompt, model, max_feature_nodes=4096, batch_size=64)`, other arguments at library defaults; free memory (`gc.collect()`, `torch.mps.empty_cache()`) after each graph, as in Experiment G. |
| Interventions | `ReplacementModel.feature_intervention` with library defaults. Record its signature and the defaults used in `results/hop/task0/metrics.json`. |

Prompts are plain text with `<bos>`, with no few-shot examples and no chat template, matching the
Neuronpedia graph. A **position** is a token index. For an answer word, the **answer token** is the
first token of the word with a leading space (` Spanish`). For a country, the **country token** is
the last token of its name.

**Feature notation.** `(l, i)` is feature `i` of the layer-`l` transcoder. `a(l, i, p)` is its
activation at position `p`.

## 2. Data (`hop/data.py`)

**Countries.** Use the rows of `middle/countries.py` (country, capital, continent, language), and
add four rows by hand with the language most commonly spoken: Mexico → Spanish, United States →
English, Canada → English, Russia → Russian. Drop any country whose language's answer token is
also another language's answer token.

**Kinds of prompt.** `{desc}` is a description that picks out one country.

| Kind | Template | Answer |
| --- | --- | --- |
| TWO | `The language most commonly spoken in {desc} is` | the language |
| HOP1 | `{Desc} is` (capitalised) | the country's first token |
| HOP2 | `The language most commonly spoken in {country} is` | the language |
| REF | the 10 reference sentences below, ending in `{country}` | none |

**Descriptions.**

- **TARGET:** `the country south of the United States` (Mexico). This is the Neuronpedia prompt.
- **CAP family:** `the country whose capital is {capital}`, one item per row of
  `middle/countries.py` (capitals there never contain the country's name).
- **DESC set**, written now and not changed after Task 1:
  `the country north of the United States` (Canada), `the country south of Canada` (United States),
  `the largest country in South America` (Brazil), `the country shaped like a boot` (Italy),
  `the country where the pyramids of Giza are` (Egypt), `the largest country in the world` (Russia),
  `the country whose capital is Mexico City` (Mexico),
  `the country directly south of the United States` (Mexico),
  `the country that borders the United States to the south` (Mexico),
  `the country with the Eiffel Tower` (France), `the country of the Acropolis` (Greece),
  `the island country where Tokyo is` (Japan).

**Reference sentences** (REF), with activations read at the country token:
`I have always wanted to visit {C}`, `The flag of {C}`, `She was born and raised in {C}`,
`The economy of {C}`, `We spent two weeks travelling around {C}`,
`The national football team of {C}`, `A map of {C}`, `The history of {C}`,
`He moved back to {C}`, `The president visited {C}`.

**Splits.** Shuffle the eligible CAP items (Task 1) with seed 0. The first 10 are DEV and the rest
are TEST. DEV is used only to fix the injection scale in Task 3. TARGET and DESC are reported
separately and never used to choose anything.

## 3. Tasks

### Hop task 0: set-up and checks

1. Download the model and transcoders. Record sizes, versions and revisions.
2. On TARGET and 5 CAP prompts, compare the next-token log-probs of the plain TransformerLens model
   with those of the ReplacementModel (with error terms, so it should reproduce the model). Record
   the max absolute difference.
3. Identity check: on the same prompts, call `feature_intervention` and set the 20 most active
   features at the last position to their own values. Record the max absolute change in
   log-probs.
4. Build the TARGET graph. Record its time, peak memory (`psutil` RSS plus
   `torch.mps.driver_allocated_memory()`), number of active features, and the library's
   replacement and completeness scores.

**Gate H0.** Step 2 difference ≤ 0.01, step 3 change ≤ 0.001, the graph builds in ≤ 20 minutes,
and peak memory ≤ 48 GB. If it fails, write `STOP.md` with the numbers.

### Hop task 1: behaviour

For every TARGET, DESC and CAP item, run TWO, HOP1 and HOP2. Record the top-5 next tokens with
probabilities, and whether the answer token is top-1.

An item is **eligible** when its TWO, HOP1 and HOP2 are all top-1 correct. For HOP1, a country
whose first token is ` the` counts as wrong.

Write `results/hop/task1/items.csv` (one row per item) and `metrics.json`: accuracy per kind and
set, and the number of eligible items per set.

**Gate H1.** TARGET is eligible, and at least 25 CAP items are eligible (10 DEV plus at least 15
TEST). If TARGET's TWO is wrong, stop: the Neuronpedia graph shows ` Spanish`, so a mismatch means
the set-up differs. If fewer than 25 CAP items are eligible, report the count and continue with
TARGET and DESC only.

### Hop task 2: country features

1. Run every REF sentence for every country (about 95 countries × 10 sentences). At the country
   token, record every feature's activation. Average over the 10 sentences to get `m_C(l, i)`, and
   count how many of the 10 sentences activate the feature (`n_C(l, i)`).
2. **Country features** of country C: features with `n_C ≥ 8` and
   `m_C ≥ 2 × max over other countries C' of m_C'`. Write the count per country and per layer.
3. **Bridge set** `M(item)`: the country features of the item's bridge country that are active
   (> 0) in the TWO prompt at any **intervention position**. The intervention positions are the
   description's last token and every later position. For CAP, the description's last token is
   the capital's last token. For TARGET, it is ` States`. Record each feature's layer and the
   positions where it fires.
4. In the TARGET graph, list the 30 feature nodes with the highest influence on the ` Spanish`
   logit. For each, give its layer, position, activation, whether it is in `M(TARGET)`, its top
   country by `m_C` and that country's selectivity ratio, and its Neuronpedia link (source set
   `gemmascope-transcoder-16k`; open one link by hand to check the URL format). Save the graph
   files with the library's export function, so John can open them in the local graph viewer.

Report the size of `M` for TARGET, DESC and CAP items. An empty `M` is a result: it means no
bridge-country feature passing this rule fires. Write it down and carry on; the decision table
deals with it.

### Hop task 3: the swap on the explicit prompt (positive control)

This task checks the method on HOP2 prompts, where the country is written in the text. If swapping
country features there cannot change the language, then a failure on TWO prompts says nothing.

For an item with bridge C, the **swap countries** are 3 countries C' chosen with seed 0 among
countries whose language differs from C's and whose own HOP2 is top-1 correct. For TARGET, add
Brazil as a fourth, fixed swap country.

**Swap on a prompt**, with source country C, swap country C', scale `s`, and a set of positions P:
- **suppress:** at every position in P, set each country feature of C that is active there to 0;
- **inject:** at every position in P, add each country feature `(l, i)` of C' whose layer is within
  ±1 of a layer that holds a country feature of C, set to `s × m_C'(l, i)`.

On HOP2 prompts, P is the country token and every later position.

1. **Scale.** On the 10 DEV items' HOP2 prompts, run swaps with `s ∈ {1, 2, 4, 8}`. Fix `s*` as the
   smallest `s` for which at least half of the (item, C') pairs flip, meaning the top-1 token
   becomes the answer token of C''s language. If no `s` does, `s* = 8`. Write `s*` to
   `results/hop/task3/metrics.json`. It is not changed afterwards.
2. **Control.** On TEST HOP2 prompts at `s*`, report the flip rate. Also run two null swaps:
   - **random inject:** suppress as above, but inject random active-at-REF features in the same
     layers, matched in count and in total injected activation;
   - **suppress only.**

   Report the flip rate and the mean change in `logp(answer of C') − logp(answer of C)`.

**Gate H2.** The TEST HOP2 flip rate at `s*` is ≥ 0.5, and the random-inject flip rate is ≤ 0.1.
If it fails, write `STOP.md`: country features found this way cannot move the answer, so the swap
on TWO prompts cannot be read.

### Hop task 4: the swap on the two-hop prompts

On every eligible TWO prompt (TARGET, DESC, CAP TEST), with P = all intervention positions and
scale `s*`:

| Condition | What it does |
| --- | --- |
| `swap` | suppress `M(item)`, inject C' (as in Task 3, using C's country-feature layers) |
| `suppress` | suppress `M(item)` only |
| `random_suppress` | set to 0 the same number of features active at the same positions and layers, not in `M`, matched in total activation; 20 random draws |
| `random_inject` | suppress `M`, inject random features as in Task 3 |
| `swap_last` | `swap`, but P = the last position only |
| `swap_desc` | `swap`, but P = the intervention positions except the last |

For each item and condition, record the top-5 tokens and the **margin**: the answer token's logit
minus the mean logit of the answer tokens of all other languages in the country table.

Report per set:
- **flip rate:** the fraction of (item, C') pairs whose top-1 becomes C''s language;
- **residual:** the suppressed margin divided by the original margin (median and IQR), next to the
  same ratio for `random_suppress`;
- flip rates for `swap_last` and `swap_desc`;
- the layer histogram of the features in `M` that were active.

Items with an empty `M` are reported as a count and left out of the rates.

### Hop task 5: where the bridge appears (logit lens and features by layer)

For TARGET and every eligible TWO prompt, at the last position and at ` States` (or at the
capital's last token), apply Gemma's final norm and unembedding to the residual stream after each
layer, keeping Gemma's final logit soft-capping. Record per layer the rank of the bridge country's
first token and of the answer token. Define `L_bridge` and `L_answer` as the first layer after
which each is in the top 10 (or "none"), as in Experiment E.

Report:
- their distributions;
- for each item, the earliest and median layer of its active `M` features;
- whether `M` features fire before `L_answer`. Give a count, with no test.

### Hop task 6: report

Write `results/hop/REPORT.md` with:

- the gate numbers;
- the Task 1 accuracy table;
- the TARGET section: the top-30 node table, `M(TARGET)`, every Task 4 condition with top-5 tokens,
  and the logit-lens ranks by layer (one plot);
- the per-set tables of Tasks 4 and 5;
- the decision-table row for TARGET, and the row for CAP TEST.

Then stop for review.

## 4. Decision table

Use flip rate F (`swap`) and median residual R (`suppress`) from Task 4. Apply it to TARGET (one
item; F is over its 4 swap countries) and to CAP TEST separately.

| Result | Reading |
| --- | --- |
| Gate H2 fails | Method failure: country features found by REF selectivity do not steer the language even when the country is written. Nothing is concluded about the hop. |
| `M` empty | No Mexico-selective feature is active, so the bridge (if any) is not a country feature this rule finds. Consistent with a shortcut, but not proof of one. |
| F ≥ 0.5 and R ≤ 0.5 | **Hop.** The bridge country is represented and carries most of the answer. |
| F ≥ 0.5 and R > 0.5 | **Hop plus shortcut.** The bridge can redirect the answer, but more than half of the margin survives without it. |
| F < 0.5 and R > 0.5 | **Shortcut.** Removing the bridge features barely matters and injecting another country does not redirect. |
| F < 0.5 and R ≤ 0.5 | **Inconclusive.** The features are needed but injection does not redirect; they may carry something other than country identity. |
| `random_suppress` residual ≤ 0.5 as well | Any R reading is void: removing that many features breaks the prompt whatever they are. |

`swap_last` against `swap_desc` says where the bridge is used: at the last token, or at the
description tokens with the answer read across by attention. Report both flip rates. The table
does not interpret them further.

## 5. Size and time (estimates, check in Task 0)

Gemma-2-2B in float32 takes about 10.5 GB. The 26 transcoders take about 8 GB in float32 (lazy
loading as in G is fine). One graph of about 17 tokens with `max_feature_nodes=4096` should take
a few minutes. REF needs about 950 forward passes, and Tasks 3–4 about 3,000 intervention passes,
at well under a second each on MPS. Expected total: 1–2 hours.
