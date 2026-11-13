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
| `paper_config.R` | Shared settings, incl. the switch `EXCLUDE_LAYPEOPLE` (see below) |
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

## Sensitivity run without laypeople and unverified users

The main analysis keeps all real replies, including the 377 (4%) from users whose flair marks them as laypeople or
not verified. To repeat everything without them, set `EXCLUDE_LAYPEOPLE=1` for every R script (settings in
`paper_config.R`; the filter is the regular expression `Layperson|not yet been verified` on the reply's `author_type`).
Results go to the separate folder `../../output/5_inference_final_expert`, so the main results are not overwritten;
`inference_results.csv` is always read from the main folder. Threads left without a real reply drop out with their AI
replies, and the response weights are recomputed from the remaining real replies. Steps 1 and 2 are not repeated.
Excluded are the 377 replies with a layperson / unverified flair plus 1 reply without any flair (378; 147 threads end up
without a real reply and are dropped with their 2,394 AI replies, i.e. 1.9% of all AI replies; in 67 further threads the
number of real replies falls and the AI replies get lower weights). `paper_descriptives.R` checks these counts and
`drop_laypeople()` stops if any lay/unverified flair is left.

```bash
export EXCLUDE_LAYPEOPLE=1
Rscript 3i_topic_llm_moderation_v2.R          # hours; fits are cached in the new folder
Rscript 3j_strategy_models.R                  # hours
SUBS_SEED=1 MECHS=ER Rscript 3j_strategy_models.R   # thinned fits, one mechanism per run (ER, IP, EX), if the robustness section is wanted
(cd ../../output/5_inference_final_expert && Rscript ../../src/5_inference_final/3k_subsampling_comparison.R && Rscript ../../src/5_inference_final/3k_subsampling_topic_shifts.R)
Rscript paper_descriptives.R
Rscript paper_joint_tests.R
Rscript -e 'rmarkdown::render("paper_empirical_section.Rmd", output_file = "paper_empirical_section_expert.pdf")'
```

The Rmd then describes the exclusion in the data section and in the limitations; all numbers come from the new folder.
The thread-level descriptives (`paper_desc_threads.csv`) then refer to the retained threads only. The responder
covariate `first_responder_roles.csv` is the collaborator's file and still describes each thread's first responder,
who may be one of the excluded users.

## Paper files

`paper_empirical_section.Rmd` (source), `.pdf` and `.docx` (outputs), `preamble_paper.tex`, `references.bib`,
`references_word.bib`, `elsevier-harvard.csl` (bibliography and layout).

## Legacy

`/legacy` holds the two pilots (`3_inference_pilot1`, `4_inference_pilot2`), the earlier frequentist and
Bayesian analyses (scripts 3, 3b to 3h of this folder with their outputs and fits), the HTML/PDF reports
(including `report_topic_moderation_v2`, the long supporting report of steps 3 and 4), and run logs.
