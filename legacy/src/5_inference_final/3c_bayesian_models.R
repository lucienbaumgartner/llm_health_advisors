#!/usr/bin/env Rscript
#
# Bayesian companion to 3b_mixed_effects_models.R, using brms/Stan instead of
# ordinal::clmm / lme4::glmer for the SAME two models per mechanism:
#
#   A. label | weights(w) ~ group + log_length + (1|post_id)            [all data]
#   B. label | weights(w) ~ model * method + log_length + (1|post_id)   [synthetic only]
#
# Response weights w (no outlier exclusion, no random subsampling): synthetic
# comments outnumber real ones per thread (synthetic was generated against each
# thread's TOTAL comment count, real_comments is a smaller clinician-only
# subset), so an unweighted fit lets long threads count for more on the
# synthetic side than on the real side. Instead of discarding data by random
# per-thread subsampling - which a 50-replicate stability check showed adds
# noticeable variance and can flip borderline contrasts (see
# report_outlier_exclusion.Rmd) - every comment is kept and each synthetic
# comment gets weight w = min(1, real_count / n_cell), where real_count is that
# thread's number of real comments and n_cell is the number of synthetic
# comments in its (thread, model, method) cell; real comments get w = 1. The
# weights sum to what random subsampling would keep (real comments + the
# min(n_cell, real_count) per cell), and the expected subsampled log-likelihood
# equals this weighted log-likelihood. brms multiplies each row's log-likelihood
# by w, so this is a pseudo-likelihood: point estimates target the same
# quantity, but posterior widths are approximate (they behave as if the sum of
# weights were the sample size) and are not corrected for the weighting. A
# frequentist check on Interpretations/Explorations (3g_weighted_vs_subsample.R)
# found the weighted estimates a median of ~0.65 SE from the 50-subsample
# average and naive SEs ~12% larger than a single subsample's, so the weighted
# fit is a different estimator, not an exact stand-in for subsampling.
# Subsampling sensitivity checks are planned separately.
#
# Motivation: on the full dataset, ER's Model B (the model x method
# interaction) fails to converge under maximum likelihood - max|grad| = 280,
# a non-positive-definite Hessian (see 3b_mixed_effects_models.R and
# report.Rmd's convergence discussion), consistent with quasi-separation in
# sparse cells (e.g. grok_dg: 2/14,258 at label 2). The standard frequentist
# fix for separation is Firth-style penalized likelihood, but there is no
# mature R (or Python) implementation that combines an ordinal cumulative-
# link outcome AND a random intercept: ordinal::clm/clmm has no bias-
# reduction option (checked directly - args() and NEWS have nothing), and
# brglm2 only covers fixed-effects GLMs (no cumulative-link/proportional-odds
# family, no random effects). A weakly-informative-prior Bayesian fit
# resolves the same separation problem the same way Firth's penalty does -
# both keep estimates finite by pulling them away from the boundary - while
# keeping the interaction + post_id random intercept structure intact.
#
# Same family choices as 3b: ER uses all 3 empathy levels -> ordinal
# cumulative("logit"); IP and EX never predict label 1 in this data (checked
# directly), so they're fit as bernoulli("logit") on I(label == 2).
#
# ############################################################################
# # THIS SCRIPT IS NOT MEANT TO BE RUN CASUALLY - DO NOT SOURCE/RUN WITHOUT  #
# # DELIBERATELY DECIDING TO. Six brms/Stan fits at this data scale (up to  #
# # 137,278 rows, a 6,600-level post_id random effect) are each likely to   #
# # take substantially longer than the equivalent clmm/glmer fits in 3b     #
# # (ER's clmm fits alone took ~2 hours EACH there). Realistic expectation: #
# # many hours to multiple days for the full set of fits, dominated by ER's #
# # cumulative-family models. Run it in the background, not interactively:  #
# #   nohup Rscript 3c_bayesian_models.R > bayesian_models.log 2>&1 &       #
# # Consider doing a quick timing probe first (see the commented-out block  #
# # near the bottom of this file) before committing to the full run.        #
# ############################################################################
#
# Priors: normal(0, 2.5) on all fixed effects ("b") and the ordinal
# thresholds ("Intercept") - the standard weakly-informative default for
# logistic-type models (Gelman, Jakulin, Pittau & Su 2008). This is the part
# that actually fixes separation: brms's own DEFAULT prior on "b" is an
# improper flat prior (confirmed via get_prior() - this is NOT already
# regularized out of the box), which would leave the same non-identifiability
# problem the frequentist fit has. student_t(3, 0, 2.5) on the post_id
# random-effect sd is brms's own default (kept explicit here for clarity).
# If real fits show these are too tight/loose (e.g. posteriors piling up
# against the prior), widen PRIOR_B_SCALE and refit - the .rds cache will
# only reuse a fit if the formula/data/prior are unchanged.
#
# Contrasts use emmeans on the *link* (log-odds) scale, exponentiated by hand
# into odds ratios - deliberately uniform across both families (rather than
# emmeans' native type="response" transform for glmer-style binomial models),
# because brms' own emmeans support for ordinal models is explicitly
# "experimental" and ignores threshold parameters for anything but plain
# linear-predictor contrasts; working on the link scale for both families
# keeps this script's treatment of ER and IP/EX identical and avoids relying
# on that experimental response-scale path. Credible intervals are highest
# posterior density (HPD) at 95%. There's no p-value in a Bayesian fit; the
# closest analogue reported here is bayestestR::p_direction() - the
# proportion of the posterior on the same side of zero as the point estimate
# (pd > 0.975 is roughly comparable to two-sided p < 0.05, per Makowski et
# al. 2019's calibration, but it's a different quantity, not a substitute -
# see the report for how this is framed).

suppressMessages({
  library(dplyr)
  library(brms)
  library(emmeans)
  library(bayestestR)
})

out_dir <- "../../output/5_inference_final"
model_dir <- file.path(out_dir, "models_bayesian")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
SYNTH_GROUPS <- as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_"))
GROUP_LEVELS <- c("real", SYNTH_GROUPS)

# ---------------------------------------------------------------------
# sampler settings - shared across all fits
# ---------------------------------------------------------------------
# Reproducibility: a fixed `seed` alone is NOT enough when within-chain
# threading (reduce_sum) is enabled - verified directly on this machine: two
# identical brm() calls with the same seed but threads=threading(2) produced
# completely different draws (max abs diff ~27 on a toy fit), because
# multi-threaded reduce_sum sums the log-likelihood in a different order
# each run, and HMC trajectories are chaotically sensitive to that
# floating-point-level difference. Threading is enabled anyway (below),
# trading bit-exact re-run reproducibility for a large speedup on ER's
# ~130k-row cumulative-family fits: the reproducibility guarantee here comes
# from `file =` below caching the complete fitted posterior (every draw) to
# disk indefinitely, not from being able to regenerate identical draws by
# rerunning the sampler. That's judged sufficient since the cached .rds is
# the actual artifact the report reads from - it doesn't get regenerated on
# every run. If the caches were ever lost and bit-exact regeneration
# mattered, set THREADS_PER_CHAIN <- 1 (verified to give bit-identical draws
# across independent runs with the same seed) before refitting.
#
# cmdstanr backend + within-chain threading, since ER's cumulative-family
# fits on ~130k rows are by far the most expensive part of this script. This
# machine has 16 cores; 4 chains x 4 threads uses all of them. Bump
# ITER/WARMUP if Rhat/ESS look marginal once real fits are in.
CHAINS <- 4
THREADS_PER_CHAIN <- 4
ITER <- 2000
WARMUP <- 1000
# Fits that needed more draws to meet the Rhat < 1.01 criterion: with 2000
# iterations the weighted IP and EX Model A fits had max Rhat = 1.011 (ESS ratio
# ~0.21, no divergences), so they are refit with 4000 iterations (same warmup).
# brm()'s `file` cache ignores changes to iter/warmup, so refitting one of these
# means moving its cached .rds aside first (done for the run recorded in
# run_metadata.txt; the 2000-iteration fits are kept in models_bayesian/superseded_iter2000/).
ITER_OVERRIDES <- c(IP_A_group_omnibus = 4000, EX_A_group_omnibus = 4000)
iter_for <- function(cache_name) if (cache_name %in% names(ITER_OVERRIDES)) ITER_OVERRIDES[[cache_name]] else ITER
SEED <- 20260101
set.seed(SEED)  # covers any R-level randomness; brms/Stan's own RNG is seeded separately via `seed=` below

PRIOR_B_SCALE <- 2.5   # sd of the normal() prior on fixed effects and thresholds (logit scale)
PRIOR_SD_SCALE <- 2.5  # scale of the student_t(3, 0, .) prior on the post_id random-effect sd

priors_for <- function(family_name) {
  p <- c(
    brms::prior_string(sprintf("normal(0, %s)", PRIOR_B_SCALE), class = "b"),
    brms::prior_string(sprintf("student_t(3, 0, %s)", PRIOR_SD_SCALE), class = "sd")
  )
  if (family_name == "cumulative") {
    p <- c(p, brms::prior_string(sprintf("normal(0, %s)", PRIOR_B_SCALE), class = "Intercept"))
  } else {
    p <- c(p, brms::prior_string(sprintf("student_t(3, 0, %s)", PRIOR_B_SCALE), class = "Intercept"))
  }
  p
}

# Full-environment record for replication: the seed only reproduces results
# under the same package/CmdStan versions (Stan's algorithms occasionally
# change between releases in ways that change draws even with an identical
# seed). Written once up front rather than per-fit since it's the same for
# the whole run.
writeLines(c(
  sprintf("date: %s", Sys.time()),
  sprintf("seed: %d", SEED),
  sprintf("chains: %d, iter: %d, warmup: %d, threads_per_chain: %d", CHAINS, ITER, WARMUP, THREADS_PER_CHAIN),
  sprintf("iteration overrides: %s", paste(names(ITER_OVERRIDES), ITER_OVERRIDES, sep = " = ", collapse = ", ")),
  sprintf("prior_b_scale: %s, prior_sd_scale: %s", PRIOR_B_SCALE, PRIOR_SD_SCALE),
  "weights: synthetic w = min(1, real_count / cell_n), real w = 1 (no outlier exclusion, no subsampling)",
  sprintf("cmdstan_version: %s", as.character(cmdstanr::cmdstan_version())),
  "", "sessionInfo():", capture.output(sessionInfo())
), file.path(model_dir, "run_metadata.txt"))

cat("Loading data...\n")
results <- read.csv(file.path(out_dir, "inference_results.csv"))
results <- results %>% filter(!is.na(response_post) & response_post != "")

results <- results %>%
  mutate(
    group = factor(ifelse(source == "real", "real", paste(model, method, sep = "_")), levels = GROUP_LEVELS),
    post_id = factor(post_id),
    length = lengths(strsplit(as.character(response_post), "\\s+")),
    log_length = log(pmax(length, 1)),
    ER_f = factor(ER_label, levels = c(0, 1, 2), ordered = TRUE),
    IP_bin = as.integer(IP_label == 2),
    EX_bin = as.integer(EX_label == 2)
  )

# Response weights (see header). A thread with no real comment would get w = 0
# and contribute nothing; none exist in this data, but such rows are dropped
# rather than passed to Stan with zero weight.
real_count <- results %>% filter(source == "real") %>% count(post_id, name = "real_count")
results <- results %>%
  left_join(real_count, by = "post_id") %>%
  mutate(real_count = coalesce(real_count, 0L)) %>%
  group_by(post_id, model, method) %>%
  mutate(cell_n = n()) %>%
  ungroup() %>%
  mutate(w = ifelse(source == "real", 1, pmin(1, real_count / cell_n))) %>%
  filter(w > 0)

synth <- results %>%
  filter(source == "synthetic") %>%
  mutate(model = factor(model, levels = MODEL_LEVELS),
         method = factor(method, levels = METHOD_LEVELS))

cat(sprintf("n = %d (all), %d (synthetic only), %d posts; sum of weights = %.0f (%.0f synthetic)\n",
            nrow(results), nrow(synth), n_distinct(results$post_id), sum(results$w), sum(synth$w)))

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

# brm()'s own `file` caching (unlike 3b's manual saveRDS/readRDS) also
# detects if the underlying data/formula/prior changed and refits
# automatically, so this doesn't need a hand-rolled fit_or_load(). The saved
# .rds is the complete brmsfit (formula, data, priors, and every posterior
# draw), so it's independently sufficient to reproduce/inspect this fit's
# results without rerunning anything.
fit_brms <- function(formula, data, family, cache_name, priors) {
  brm(
    formula = formula,
    data = data,
    family = family,
    prior = priors,
    chains = CHAINS,
    cores = CHAINS,
    threads = if (THREADS_PER_CHAIN > 1) threading(THREADS_PER_CHAIN) else NULL,
    backend = "cmdstanr",
    iter = iter_for(cache_name),
    warmup = WARMUP,
    seed = SEED,
    file = file.path(model_dir, cache_name),
    file_refit = "on_change"
  )
}

check_convergence_bayes <- function(fit) {
  rhats <- brms::rhat(fit)
  neff_ratios <- brms::neff_ratio(fit)
  divergent <- tryCatch(sum(subset(brms::nuts_params(fit), Parameter == "divergent__")$Value),
                         error = function(e) NA_integer_)
  max_rhat <- suppressWarnings(max(rhats, na.rm = TRUE))
  min_neff_ratio <- suppressWarnings(min(neff_ratios, na.rm = TRUE))
  ok <- is.finite(max_rhat) && max_rhat < 1.01 && !is.na(divergent) && divergent == 0 && min_neff_ratio > 0.1
  msg <- if (ok) {
    NA_character_
  } else {
    sprintf("max Rhat = %.3f, min ESS ratio = %.3f, %s divergent transition(s)",
            max_rhat, min_neff_ratio, ifelse(is.na(divergent), "unknown", divergent))
  }
  data.frame(converged = ok, max_rhat = max_rhat, min_neff_ratio = min_neff_ratio,
             divergent_transitions = divergent, message = msg)
}

# Tidy an emmeans contrast on a brmsfit into the same shape 3b's
# tidy_contrast() produces, for the report to read as similarly as possible:
# contrast (+ optional method/model conditioning column), odds_ratio,
# OR_LCL, OR_UCL, plus pd (probability of direction) in place of p_value -
# see the module docstring for why both families are read on the link scale
# and exponentiated by hand rather than using emmeans' response-scale option.
tidy_contrast_bayes <- function(ct, mechanism) {
  s <- as.data.frame(summary(ct, point.est = median))
  s$odds_ratio <- exp(s$estimate)
  s$OR_LCL <- exp(s$lower.HPD)
  s$OR_UCL <- exp(s$upper.HPD)
  pd <- as.data.frame(bayestestR::p_direction(ct))
  # merge on whatever grouping columns both frames share (contrast, and
  # method/model when present) rather than assuming row order matches
  join_cols <- intersect(names(s), names(pd))
  s <- s %>% left_join(pd, by = join_cols)
  s$mechanism <- mechanism
  keep_cols <- intersect(c("contrast", "method", "model", "odds_ratio", "OR_LCL", "OR_UCL", "pd", "mechanism"), names(s))
  s[, keep_cols]
}

fit_mechanism_bayes <- function(mechanism, response_col, family_name) {
  cat(sprintf("\n=== %s ===\n", mechanism))
  family_obj <- if (family_name == "cumulative") cumulative("logit") else bernoulli("logit")
  priors <- priors_for(family_name)

  # --- Model A: group (real + 9 combos), all data ---
  cat(sprintf("--- %s: Model A (group, n=%d) ---\n", mechanism, nrow(results)))
  fA_formula <- as.formula(sprintf("%s | weights(w) ~ group + log_length + (1|post_id)", response_col))
  fA <- fit_brms(fA_formula, results, family_obj, sprintf("%s_A_group_omnibus", mechanism), priors)
  convA <- check_convergence_bayes(fA)
  cat(sprintf("  converged: %s%s\n", convA$converged, ifelse(is.na(convA$message), "", paste0(" - ", convA$message))))

  length_row <- as.data.frame(brms::fixef(fA, probs = c(0.025, 0.975)))
  length_row <- length_row[rownames(length_row) == "log_length", , drop = FALSE]
  pd_length <- as.numeric(bayestestR::p_direction(fA, parameters = "^b_log_length$")$pd)
  length_effect <- data.frame(mechanism = mechanism, log_length_coef = length_row$Estimate,
                               CI_LCL = length_row$Q2.5, CI_UCL = length_row$Q97.5, pd = pd_length)

  emA <- emmeans(fA, ~group)
  ctA <- contrast(emA, "trt.vs.ctrl")
  vs_real <- tidy_contrast_bayes(ctA, mechanism)

  # --- Model B: model * method, synthetic only ---
  cat(sprintf("--- %s: Model B (model*method, n=%d) ---\n", mechanism, nrow(synth)))
  fB_formula <- as.formula(sprintf("%s | weights(w) ~ model * method + log_length + (1|post_id)", response_col))
  fB <- fit_brms(fB_formula, synth, family_obj, sprintf("%s_B_model_x_method", mechanism), priors)
  convB <- check_convergence_bayes(fB)
  cat(sprintf("  converged: %s%s\n", convB$converged, ifelse(is.na(convB$message), "", paste0(" - ", convB$message))))

  # Simple-effects contrasts (same rationale as 3b: the interaction may be
  # credible for some mechanisms, which would make pooled main-effect means
  # misleading) - between models within each method, and between methods
  # within each model. Unlike the frequentist version, this doesn't need a
  # try/catch fallback for a non-positive-definite Hessian: HMC sampling
  # from a proper (proper-prior) posterior doesn't have that failure mode -
  # that's the whole point of trying this.
  em_by_method <- emmeans(fB, ~ model | method)
  ct_between_model <- contrast(em_by_method, "pairwise")
  between_model <- tidy_contrast_bayes(ct_between_model, mechanism)

  em_by_model <- emmeans(fB, ~ method | model)
  ct_within_model <- contrast(em_by_model, "pairwise")
  within_model <- tidy_contrast_bayes(ct_within_model, mechanism)

  list(
    convergence = bind_rows(
      cbind(mechanism = mechanism, model_stage = "A_group_omnibus", model_type = family_name, convA),
      cbind(mechanism = mechanism, model_stage = "B_model_x_method", model_type = family_name, convB)
    ),
    length_effect = length_effect,
    vs_real = vs_real,
    between_model = between_model,
    within_model = within_model
  )
}

res_er <- fit_mechanism_bayes("ER", "ER_f", "cumulative")
res_ip <- fit_mechanism_bayes("IP", "IP_bin", "bernoulli")
res_ex <- fit_mechanism_bayes("EX", "EX_bin", "bernoulli")

# --- Robustness check (IP only), matching 3b's fA_nolen: does dropping
# log_length from Model A change the grok/openai vs.-real picture? ---
cat("\n=== IP: Model A without log_length (robustness check) ===\n")
fA_nolen <- fit_brms(IP_bin | weights(w) ~ group + (1|post_id), results, bernoulli("logit"),
                      "IP_A_nolength", priors_for("bernoulli"))
em_nolen <- emmeans(fA_nolen, ~group)
ct_nolen <- contrast(em_nolen, "trt.vs.ctrl")
vs_real_nolength <- tidy_contrast_bayes(ct_nolen, "IP")
write.csv(vs_real_nolength, file.path(out_dir, "bayesian_contrasts_vs_real_nolength_IP.csv"), row.names = FALSE)
print(vs_real_nolength %>% filter(grepl("grok|openai", contrast)))

all_convergence <- bind_rows(res_er$convergence, res_ip$convergence, res_ex$convergence)
all_length_effect <- bind_rows(res_er$length_effect, res_ip$length_effect, res_ex$length_effect)
all_vs_real <- bind_rows(res_er$vs_real, res_ip$vs_real, res_ex$vs_real)
all_between_model <- bind_rows(res_er$between_model, res_ip$between_model, res_ex$between_model)
all_within_model <- bind_rows(res_er$within_model, res_ip$within_model, res_ex$within_model)

# No BH correction here - there's no p-value family to correct; pd is a
# per-contrast posterior summary, not a frequentist test statistic. The
# report instead flags contrasts using both a pd threshold (> 0.975, the
# rough two-sided-p<0.05 analogue) and "does the 95% CrI exclude an odds
# ratio of 1", which will usually agree.

write.csv(all_convergence, file.path(out_dir, "bayesian_convergence.csv"), row.names = FALSE)
write.csv(all_length_effect, file.path(out_dir, "bayesian_length_effect.csv"), row.names = FALSE)
write.csv(all_vs_real, file.path(out_dir, "bayesian_contrasts_vs_real.csv"), row.names = FALSE)
write.csv(all_between_model, file.path(out_dir, "bayesian_contrasts_between_model.csv"), row.names = FALSE)
write.csv(all_within_model, file.path(out_dir, "bayesian_contrasts_within_model.csv"), row.names = FALSE)

cat("\n=== Convergence ===\n")
print(all_convergence)
cat("\n=== Contrasts vs. real (Model A) ===\n")
print(all_vs_real)
cat("\n=== Between-model, within-method contrasts (Model B) ===\n")
print(all_between_model)
cat("\n=== Within-model, between-method contrasts (Model B) ===\n")
print(all_within_model)
cat("\n=== log(length) coefficient (Model A) ===\n")
print(all_length_effect)

cat("\nWrote bayesian_{convergence,length_effect,contrasts_vs_real,contrasts_between_model,contrasts_within_model}.csv to", out_dir, "\n")

# ---------------------------------------------------------------------
# Optional: quick timing probe before committing to the full run above.
# Uncomment and run on its own (comment out everything above from
# `res_er <- fit_mechanism_bayes(...)` onward first, or just run this block
# interactively) to get a realistic per-iteration time estimate on THIS
# machine with THIS data's actual size, rather than guessing from the toy
# benchmark used to validate this script's emmeans/contrast code during
# development (800 rows, 60-level cluster, 300 iterations, ~7s - not
# representative of 137k rows / 6,600 levels).
#
# probe <- fit_brms(ER_f | weights(w) ~ group + log_length + (1|post_id), results,
#                    cumulative("logit"), "ER_A_TIMING_PROBE",
#                    priors_for("cumulative"))
# (the probe still runs the full CHAINS/ITER/WARMUP above - to make it
# actually quick, temporarily lower ITER/WARMUP or pass iter=/warmup=
# directly to fit_brms's underlying brm() call instead)
