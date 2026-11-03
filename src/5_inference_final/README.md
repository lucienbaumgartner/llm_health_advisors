# 5_inference_final — Real vs. Synthetic Empathy (Full Dataset)

Same pipeline as [`../4_inference_pilot2`](../4_inference_pilot2), run on the
full dataset (6,600 threads vs. pilot 2's 3,000) once it became available.
Same three synthetic models, same three prompting methods, same modeling
approach — this run mainly exists to see whether pilot 2's findings hold up
at scale, not to change the method.

## Input

`../../input/calls_final/analysis_dataset_6600.jsonl.gz` — one JSON object
per line, one line per Reddit r/AskDocs thread (6,600 threads), identical
schema to pilot 2's dataset (confirmed by direct structural comparison,
including that pilot 2's first record appears verbatim here — this dataset
looks like a superset of pilot 2's, not an independent draw):
- `submission_id`, `title`, `selftext`, `n_comments`, `permalink`, `created_utc`
- `real_comments`: `[{body, author_flair_text, score}, ...]`
- `synthetic`: `{model: {method: [{body, author_type}, ...]}}`, for
  `model` in **gemini**, **openai**, **grok** and `method` in **dg**
  (Discourse Generation), **sag** (Single Advice Generation), **sagii**
  (SAG + Incremental Info).

## Pipeline

### 1. `1_prepare_data.py`

Reads the gzipped JSONL into `posts.json` plus a QC report
(`qc_comment_counts.csv`) checking real vs. synthetic comment counts per
post/model/method.

**Unlike pilot 1, this isn't a bug and nothing is dropped.** Synthetic
comment counts don't match `real_comments` counts in ~26% of (post, model,
method) combinations, but that's because synthetic comments were generated
1:1 against each thread's *total* comment count (`n_comments` — matches in
99.98% of combos here), while `real_comments` is a smaller, already-filtered
subset of that thread — the same pattern pilot 2 found.

```bash
python3 1_prepare_data.py
```

### 2. `2_run_inference.py`

Same batched-inference approach as pilot 1/2. Classifies 137,281 comments
(8,625 real + 128,656 synthetic) on Apple Silicon MPS at `--batch_size 64`.

```bash
python3 2_run_inference.py
```

### 3. `3_analyze_empathy_differences.py`

Mann-Whitney U real-vs-each-(model,method)-group (27 pairwise tests) with BH
correction and rank-biserial effect sizes, plus Spearman length correlation
— → `empathy_*.csv`, `comment_length_by_group.csv`. 3 comments with an
empty/missing body are dropped before analysis (137,278 comments analyzed).

```bash
python3 3_analyze_empathy_differences.py
```

### 3b. `3b_mixed_effects_models.R`

Same two-model design as pilot 2 (Model A: `label ~ group + log(length) +
(1|post_id)`, all data; Model B: `label ~ model * method + log(length) +
(1|post_id)`, synthetic only), same ordinal `clmm` for Emotional Reactions
and binomial `glmer` for Interpretations/Explorations (still 0 occurrences
of label 1 for both, checked directly on the full 137,278-comment dataset).

**This run surfaced a real convergence failure pilot 2 didn't hit at its
smaller scale**, and the script has been hardened accordingly:

- ER's Model A converges adequately (small residual gradient, `max|grad| =
  0.0147`) — its omnibus test and vs.-real contrasts are reported normally.
- **ER's Model B (the model × method interaction) fails to converge
  outright** (`max|grad| = 280`, and its Hessian is not positive definite —
  `vcov()` errors, confirmed directly on the fitted object). This isn't the
  mild "small gradient" caveat pilot 2 reported for ER; there is no valid
  standard error or p-value obtainable from this fit at all. Consistent with
  the separation risk already known from pilot 2: `grok_dg` is the sparsest
  (model, method) cell for label 2 in this run (2/14,258), and the
  larger `post_id` random effect (6,600 vs. 3,000 levels) plausibly makes
  the interaction model's likelihood surface harder to optimize.
- Because of this, `anova_table()` (the function that falls back to a manual
  LRT when `car::Anova`'s Wald test fails) now:
  - only attempts an LRT for a term when no other requested term still
    contains it (dropping `model` while `model:method` remains isn't a
    valid marginal test — it's silently a different hypothesis — so `model`
    and `method` are reported as **not estimable** rather than substituting
    an inequivalent test);
  - catches the case where the LRT refit itself fails against an unusable
    full-model fit (which happened here for `model:method`), reporting
    **not estimable** instead of crashing.
  - The `emmeans`-based simple-effects contrasts (between-model,
    within-model) are similarly wrapped so a fit with an unusable Hessian
    is simply omitted from those two tables (with a note in the report)
    rather than producing numbers from an unreliable fit.

IP and EX converged cleanly for both models, as in pilot 2.

Every fitted model is cached to `../../output/5_inference_final/models/*.rds`.
At this dataset's scale, ER's two `clmm` fits are considerably slower than
pilot 2's ~20–35 minutes combined — Model A alone took ~1h56m and Model B
~2h14m in this run (up from a ~3,000-post to a 6,600-post random effect).
Re-running the script after an output-side change (a new contrast, a table
tweak) reuses the cached fits and finishes in seconds; delete a `.rds` file
to force a refit.

```bash
Rscript 3b_mixed_effects_models.R
```

(Needs the `ordinal`, `lme4`, `car`, and `emmeans` R packages.)

### 4. `report.Rmd`

Same structure as pilot 2's report — every number computed inline from the
data (no findings hardcoded from pilot 2 are assumed to hold; the report
recomputes which groups are "exceptions," which main effects drop out, and
which convergence outcome applies from the actual fitted results each time
it knits). The convergence-status prose distinguishes ER's two very
different outcomes above (Model A usable, Model B not) rather than
collapsing them into one caveat.

```bash
Rscript -e 'rmarkdown::render("report.Rmd", output_format = "bookdown::pdf_document2")'
Rscript -e 'rmarkdown::render("report.Rmd", output_format = "bookdown::html_document2")'
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 transformers==5.15.0 pandas scipy numpy tqdm requests filelock
```

(Same torch/transformers versions as `../2_training` and the earlier
pipelines, since the checkpoints need to be loaded with a matching version.
PDF rendering needs a LaTeX distribution with `xelatex`, e.g. TinyTeX. Both
formats need the `bookdown` R package; script 3b needs `ordinal`, `lme4`,
`car`, and `emmeans`.)

## Results

137,278 comments classified (8,625 real, 128,653 synthetic; ~14,295 per
model/method group on average). The pattern from pilot 1/2 replicates at
scale: real comments score significantly higher than **every** (model,
method) combination on all three mechanisms except Interpretations, where
**Grok scores significantly higher than real across all three prompting
strategies** (OR ≈ 2.0–2.5, BH-adjusted p mostly < 1e-30) once comment
length is controlled for — Gemini and OpenAI still score lower than real on
Interpretations. The raw Mann-Whitney picture is muddier for three of these
groups (`grok_dg`, `openai_sag`, `openai_sagii` flip direction once
`log(length)` is dropped from the model), a length confound: comment length
differs systematically across models and is a strong predictor of this
label (longer comments score higher on both Emotional Reactions and
Interpretations, but *lower* on Explorations).

The model × method interaction is significant for both mechanisms it could
be tested for (Interpretations, Explorations); it could not be tested for
Emotional Reactions, whose Model B fit failed to converge (see above) — so
ER is omitted from the simple-effects contrast tables entirely rather than
reporting numbers from that fit. Among the mechanisms where it's estimable,
the largest between-model gap is Explorations' Gemini vs. Grok within DG
(OR ≈ 8.0); the largest within-model gap is Explorations' DG vs. SAG within
Grok (OR ≈ 0.38).

See `report.pdf` / `report.html` for the full breakdown, including the
convergence, omnibus, vs.-real, and simple-effects tables, and the explicit
caveats on what is and isn't estimable for Emotional Reactions' Model B in
this run.
