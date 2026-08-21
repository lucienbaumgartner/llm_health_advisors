# 4_inference_pilot2 — Real vs. Synthetic Empathy (Pilot 2)

Same idea as [`../3_inference_pilot1`](../3_inference_pilot1), run on a much
larger pilot-2 dataset with three synthetic models instead of one, and
extended with a per-method / per-model breakdown.

## Input

`../../input/analysis_dataset.jsonl.gz` — one JSON object per line, one line
per Reddit r/AskDocs thread (3,000 threads):
- `submission_id`, `title`, `selftext`, `n_comments`, `permalink`, `created_utc`
- `real_comments`: `[{body, author_flair_text, score}, ...]`
- `synthetic`: `{model: {method: [{body, author_type}, ...]}}`, for
  `model` in **gemini**, **openai**, **grok** and `method` in **dg**
  (Discourse Generation), **sag** (Single Advice Generation), **sagii**
  (SAG + Incremental Info) — the same three prompting methods as pilot 1,
  now crossed with three models instead of one.

## Pipeline

### 1. `1_prepare_data.py`

Reads the gzipped JSONL into the same `posts.json` shape
`2_run_inference.py` expects, and writes a QC report
(`qc_comment_counts.csv`) checking real vs. synthetic comment counts per
post/model/method.

**Unlike pilot 1, this isn't a bug and nothing is dropped.** Synthetic
comment counts don't match `real_comments` counts in ~25% of
(post, model, method) combinations, but that's because synthetic comments
were generated 1:1 against each thread's *total* comment count (`n_comments`
— matches in 100% of combos), while `real_comments` is a smaller,
already-filtered subset of that thread. Confirmed by checking both
correlations directly — see the script's docstring.

```bash
python3 1_prepare_data.py
```

### 2. `2_run_inference.py`

Same batched-inference approach as pilot 1 (see that script's docstring for
why it doesn't reuse `../2_training/empathy_classifier.py`), extended for
the extra `model` dimension. Classifies ~62k comments (3,882 real + ~58k
synthetic); takes roughly 15 minutes on Apple Silicon MPS at
`--batch_size 64`.

```bash
python3 2_run_inference.py
```

### 3. `3_analyze_empathy_differences.py`

Mann-Whitney U real-vs-each-(model,method)-group (27 pairwise tests) with BH
correction and rank-biserial effect sizes, plus Spearman length correlation
— → `empathy_*.csv`, `comment_length_by_group.csv`. 4 comments with an
empty/missing body (a handful of blank synthetic generations — see
`qc_comment_counts.csv`) are dropped before analysis.

The Kruskal-Wallis omnibus test and the by-method/by-model pooled
comparisons that used to live here have been replaced by proper
mixed-effects models — see script 3b below. Those nonparametric tests
treated every comment as independent, which they aren't (multiple comments
per Reddit thread), and couldn't control for comment length directly.

```bash
python3 3_analyze_empathy_differences.py
```

### 3b. `3b_mixed_effects_models.R`

Two mixed-effects models per mechanism, both with a `post_id` random
intercept (accounting for comments clustering by thread) and `log(length)`
as a covariate:
- **A — omnibus** (`label ~ group + log(length) + (1|post_id)`, all
  comments): replaces the Kruskal-Wallis test, plus `emmeans`-based
  contrasts of each (model, method) group against real (odds ratios) — a
  model-adjusted complement to the Mann-Whitney table in script 3, not a
  replacement for it.
- **B — model × method** (`label ~ model * method + log(length) + (1|post_id)`,
  synthetic comments only): tests the model main effect, method main effect,
  and their interaction. The interaction is significant for all three
  mechanisms, which makes pooled by-method/by-model marginal means
  misleading (they average over the very factor that matters) — so instead
  of pooled means, this fits two sets of **simple-effects contrasts**:
  models compared pairwise within each method, and methods compared
  pairwise within each model.

Emotional Reactions uses all 3 empathy levels → `ordinal::clmm`.
Interpretations and Explorations never predict label 1 anywhere in this data
(0/62,161, checked directly) — a 3-level ordinal model would be degenerate
for them, so they're fit as binomial `lme4::glmer` on P(label = 2) instead.
Effect tests are Wald Type II chi-square (`car::Anova`) from a single fit,
with a manual LRT fallback if that ever fails on a given model. Convergence
is checked and reported per fit — **Emotional Reactions' models don't fully
converge** (small residual gradient, ~0.001–0.009), consistent with a real
separation risk already identified in the data (`grok`/`sagii` has 0/6,475
comments at label 2); treat ER's model-based results as directional rather
than final. Interpretations and Explorations converge cleanly.

All contrasts are reported as odds ratios. `glmer` gives these natively
(`emmeans(..., type = "response")`); `clmm` has no single "response" scale
for a 3-category cumulative-link model, so its latent-scale (log-odds)
contrast is exponentiated by hand — the same quantity a proportional-odds
model is normally reported with, just not something `emmeans` transforms
automatically for that model class.

Every fitted model is cached to `../../output/4_inference_pilot2/models/*.rds`.
`clmm` on the full ~62k-row / ~3,000-post dataset is slow (Emotional
Reactions' two `clmm` fits take ~20–35 min combined; Interpretations/
Explorations fit in well under a minute each via `glmer`) — after the first
run, re-running the script (e.g. after a change to a *table*, not the model
formula) reuses the cached fits and finishes in seconds. Delete a `.rds`
file (or the whole `models/` directory) to force a refit.

```bash
Rscript 3b_mixed_effects_models.R
```

(Needs the `ordinal`, `lme4`, `car`, and `emmeans` R packages.)

### 4. `report.Rmd`

Same structure as pilot 1's report (ordinal-labels note, mean + proportion
plots, Mann-Whitney table, comment-length section, every number computed
inline from the data), plus a "Mixed-effects models" section presenting the
convergence, omnibus, vs.-real, and model×method/simple-effects tables from
script 3b (convergence caveat front and center, not buried), and a short
note citing a coauthor's independent question-mark-based cross-check of the
Explorations finding
([`5_exploratory_analysis.ipynb`](https://github.com/cmstewart/llm_health_advisors/blob/main/Piloting/Round%205/5_exploratory_analysis.ipynb)).

Note: Table 1 (Mann-Whitney) highlights negative rank-biserial-r cells via
`kableExtra::cell_spec()` background colors. Adding the *same* technique to
more than one of the new mixed-model tables broke the PDF build
(`! Misplaced \noalign` — a known pandoc/kableExtra/xcolor conflict that
surfaces once several distinct `\cellcolor` colors appear across a
document); isolated by bisection, not fixed. The new tables are plain
(no cell shading) as a result — significance is still visible via the
`p (BH-adj.)` column.

Renders to both PDF and HTML from the same source via `bookdown`, which
gives figure/table cross-references (`Fig. \@ref(fig:...)`,
`Table \@ref(tab:...)`) that resolve correctly in *both* formats — plain
LaTeX `\label{}`/`\ref{}` only work in the PDF (LaTeX) build and show up as
broken literal text in HTML.

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

(Same torch/transformers versions as `../2_training` and
`../3_inference_pilot1`, since the checkpoints need to be loaded with a
matching version. PDF rendering needs a LaTeX distribution with `xelatex`,
e.g. TinyTeX. Both formats need the `bookdown` R package
(`Rscript -e 'install.packages("bookdown")'`); script 3b needs `ordinal`,
`lme4`, `car`, and `emmeans`.)

## Results

62,161 comments classified (3,882 real, 58,279 synthetic). Real comments
score higher than **every** (model, method) combination on all three
mechanisms except Interpretations, where openai and grok actually score
*higher* than real (p_adj ≈ 0.001–0.02) while gemini scores lower — the only
mechanism/model pairing that goes the other way. Emotional Reactions and
Explorations are unambiguous: all 9 synthetic groups score significantly
lower than real (BH-adjusted p mostly < 1e-40), consistent with pilot 1's
finding on a much smaller sample. Effect sizes remain modest throughout
(rank-biserial r ≈ -0.02–0.18) despite the very small p-values, which mostly
reflects statistical power from the large n rather than a stark separation.

The mixed-effects models (accounting for post clustering, controlling for
comment length) confirm the group differences (Model A's `group` term
significant for all three mechanisms) and find a significant model×method
interaction for **all three mechanisms** in Model B — the effect of a
prompting method depends on which model generated it. Interpretations'
method main effect drops out once model is accounted for (p ≈ 0.45). Model A's
vs.-real contrasts mostly agree with the raw Mann-Whitney direction, with
one exception: Interpretations' grok/dg, grok/sag, and grok/sagii all score
significantly *higher* odds of strong empathy than real (OR ≈ 2.1–2.8),
consistent with — and more precisely quantified than — the raw picture.
Since the interaction rules out trusting pooled by-method/by-model means,
Model B instead reports simple-effects contrasts: the largest between-model
gap is Emotional Reactions' gemini vs. grok within SAGII (OR ≈ 18.3); the
largest within-model gap is Emotional Reactions' DG vs. SAG within openai
(OR ≈ 0.36). Emotional Reactions' models didn't fully converge — read its
model-based numbers as directional.

See `report.pdf` / `report.html` for the full breakdown, including the
mixed-effects convergence, omnibus, vs.-real, and simple-effects tables.
