# 5_inference_final — pipeline behind the paper

Everything needed to reproduce `paper_empirical_section.Rmd` is in this folder and the folders it reads.
Earlier analyses, pilots and reports were moved to `/legacy` (same relative paths as before).
Run all steps from this folder unless stated otherwise.

## Inputs

| File | Content |
|---|---|
| `../../input/calls_final/analysis_dataset_6600.jsonl.gz` | 6,600 r/AskDocs threads with real replies and the AI replies (3 models x 3 strategies) |
| `../../input/topics/topic_labels.csv` | Discipline and cancer labels (Gemini 2.5 Pro; collaborator repo, see `SOURCE.txt`) |
| `../../input/covariates/*.csv` | Severity, poster demographics, responder roles (collaborator repo, see `SOURCE.txt`) |
| `../../output/2_training/reddit_{ER,IP,EX}.pth` | Trained EPITOME classifiers (training code: `../2_training`) |
| `../../Readme.md` | Corpus description; the paper reads the corpus size from it (corpus built by `../1_preprocessing`) |

## Steps

| Step | Command | Writes (to `../../output/5_inference_final`) |
|---|---|---|
| 1 Prepare | `python3 1_prepare_data.py` | `posts.json`, `qc_comment_counts.csv` |
| 2 Classify | `python3 2_run_inference.py` | `inference_results.csv` (137,281 replies, labels ER/IP/EX) |
| 3 Topic, severity and model-by-topic models (v2) | `Rscript 3i_topic_llm_moderation_v2.R` | `v2_*.csv`, `models_bayesian/v2_*.rds` |
| 4 Strategy models (v3), weighted | `Rscript 3j_strategy_models.R` | `v3_*.csv`, `models_bayesian/v3_*.rds` |
| 5 Strategy models, thinned (robustness) | see header of `3j_strategy_models.R` (`SUBS_SEED=1`, one mechanism per run) | `v3sub1_<MECH>_*.csv`, `models_bayesian/sub1_v3_*.rds` |
| 6 Weighted vs. thinned | from `../../output/5_inference_final`: `Rscript ../../src/5_inference_final/3k_subsampling_comparison.R` and `3k_subsampling_topic_shifts.R` | `v3sub1_comparison_group_effects.csv`, `v3sub1_topic_shift_comparison.csv` |
| 7 Descriptives | `Rscript paper_descriptives.R` (reads the 250 MB `inference_results.csv` once) | `paper_desc_*.csv` |
| 8 Joint tests | `Rscript paper_joint_tests.R` (about 40 s; needs the `.rds` fits) | `paper_joint_tests.csv`, `paper_interaction_pairs.csv` |
| 9 Paper | `Rscript -e 'rmarkdown::render("paper_empirical_section.Rmd")'` (about 3 s) | `paper_empirical_section.pdf` |

Steps 3 to 5 take many hours (brms/cmdstanr, 4 chains); finished fits are cached as `.rds` in
`models_bayesian/` and reused (`file_refit = "on_change"`), so rerunning with the cache is fast.
The `.rds` files and `inference_results.csv` are not under version control (`.gitignore`).
The paper reads only small CSV files, so step 9 is fast; it stops with an error if a claim in the text
is contradicted by the results (claims check in the setup chunk).

## Paper files

`paper_empirical_section.Rmd` (source), `.pdf` and `.docx` (outputs), `preamble_paper.tex`, `references.bib`,
`references_word.bib`, `elsevier-harvard.csl` (bibliography and layout).

## Legacy

`/legacy` holds the two pilots (`3_inference_pilot1`, `4_inference_pilot2`), the earlier frequentist and
Bayesian analyses (scripts 3, 3b to 3h of this folder with their outputs and fits), the HTML/PDF reports
(including `report_topic_moderation_v2`, the long supporting report of steps 3 and 4), and run logs.
