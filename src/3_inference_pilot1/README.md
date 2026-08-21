# 3_inference — Real vs. Synthetic Empathy

Runs the empathy classifiers trained in [`../2_training`](../2_training) on the
clinician-comment inference set in [`../../input`](../../input), and compares
predicted empathy between real and synthetic comments.

## Input

`../../input/round3_clinician_synthetic_comments.xlsx` — one sheet per Reddit
post (100 sheets), each containing:
- the opening post (title + body),
- a "Real breakdown" line stating how many real clinician/layperson comments
  the post had,
- the real comment(s),
- three sets of synthetic comments, one per prompting strategy: **DG**
  (Discourse Generation), **SAG** (Single Advice Generation), **SAGII** (SAG +
  Incremental Info).

## Pipeline

### 1. `1_xlsx_to_json.py`

Parses every sheet into `output/3_inference/posts.json` and checks that the
number of real comments matches the number of synthetic comments per method
(they're generated 1:1 against the stated real-comment count).

**21/100 sheets fail this check** — the real section has fewer comment cells
than the stated breakdown total, i.e. multiple real comments were merged into
one cell upstream, in the source spreadsheet. Splitting on blank lines only
reconciles the count for 1 of the 21 (by coincidence), so this is **flagged,
not auto-fixed** — see `output/3_inference/qc_comment_counts.csv` for the
full list. If you can get a corrected version of the spreadsheet, or want a
different splitting heuristic, this is the place to change it.

```bash
python3 1_xlsx_to_json.py
```

### 2. `2_run_inference.py`

Loads the three checkpoints from `../2_training` (`reddit_ER.pth`,
`reddit_IP.pth`, `reddit_EX.pth`) and classifies every real and synthetic
comment (posts themselves are only used as the seeker-post context, not
classified). Writes `output/3_inference/inference_results.csv` with predicted
labels (0/1/2) and class probabilities for each mechanism.

Note: this reimplements batched inference rather than reusing
`../2_training/empathy_classifier.py`'s `EmpathyClassifier.predict_empathy()` —
that method loops over DataLoader batches but returns only the *last* batch's
predictions. Harmless in the original repo, since `test.py` only ever calls it
one example at a time, but wrong for batched inference over many comments.

```bash
python3 2_run_inference.py
```

### 3. `3_analyze_empathy_differences.py`

Compares predicted empathy between real comments and each synthetic method,
per mechanism: group summary stats, Kruskal-Wallis omnibus test, and
Mann-Whitney U (real vs. each method) with Benjamini-Hochberg FDR correction
and rank-biserial effect sizes. Also checks whether comment length (word
count) correlates with predicted empathy (Spearman), overall and within each
group, since real and synthetic comments turn out to differ a lot in length
and this could confound the group comparisons. Writes `empathy_summary_stats.csv`,
`empathy_group_comparison.csv`, `empathy_kruskal.csv`,
`empathy_length_correlation.csv`, `comment_length_by_group.csv` to
`output/3_inference`.

**Restricted to the 79/100 sheets where real and synthetic comment counts
actually match** — the other 21 sheets have real comments merged into fewer
cells than their stated count (see script 1), which would bias the
real-comment group, so they're dropped entirely rather than left in with
mismatched counts.

```bash
python3 3_analyze_empathy_differences.py
```

### 4. `report.Rmd`

Reads the CSVs above and produces a short report with a summary table (mean
predicted empathy by group; no median — not informative for a 3-level ordinal
label this skewed), a bar chart, a proportion breakdown of the ordinal labels
(the mean treats 0/1/2 as equally spaced, which the tests don't assume — see
the note in the report), the test results, a comment-length section
(table + scatter/trend plot + correlation tests), and a discussion — every
number in the prose is computed inline from the data (group labels, sheet
counts, percentages, fold-differences, significance counts, etc.), not
hardcoded, so the text stays correct if the underlying data changes.

Renders to both HTML and PDF (via LaTeX/xelatex):

```bash
Rscript -e 'rmarkdown::render("report.Rmd", output_format = "html_document")'
Rscript -e 'rmarkdown::render("report.Rmd", output_format = "pdf_document")'
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.13.0 transformers==5.15.0 openpyxl pandas scipy numpy tqdm requests filelock
```

(Same torch/transformers versions as `../2_training`, since the checkpoints
need to be loaded with a matching version.)

(PDF rendering needs a LaTeX distribution with `xelatex`, e.g. TinyTeX:
`Rscript -e 'tinytex::install_tinytex()'`.)

## Results (round3_clinician_synthetic_comments.xlsx, matched-count sheets only)

348 comments classified (87 real, 261 synthetic — 87 per method, from the
79/100 sheets with matching real/synthetic counts). Real comments scored
higher on **Explorations** than all three synthetic methods (Kruskal-Wallis
p ≈ 3.2e-4; all three pairwise comparisons significant after BH correction,
~4.6x higher on average). **Emotional Reactions** and **Interpretations**
show no significant real-vs-synthetic differences after correction — at this
n (87 real comments), the analysis is underpowered to rule out smaller
effects there. Effect sizes throughout are modest (rank-biserial r ≈
-0.01–0.15).

Real comments are also somewhat longer and more variable in length (mean ~42
words, SD ~44) than synthetic ones (~34-38 words, SD ~7-9 regardless of
method). Comment length is not a strong general driver of predicted empathy —
no pooled ("overall") correlation is significant after correction — so the
Explorations gap doesn't look like a simple length artifact.

See `report.html` / `report.pdf` for the full breakdown and discussion.
