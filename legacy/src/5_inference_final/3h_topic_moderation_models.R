#!/usr/bin/env Rscript
#
# Topic moderation of the real-vs-synthetic empathy gap, Bayesian (brms).
# New models alongside 3c's A/B fits - those are untouched (new cache names,
# new output files prefixed `topic_`).
#
# Topic is a THREAD-level label (one per opening post, from the collaborator's
# classify_topics.py; input/topics/topic_labels.csv, key submission_id), so
# every real and synthetic comment in a thread carries the thread's topic. The
# question is whether the gap between real and synthetic comments differs by
# topic; it is identified within thread, which the post_id intercept protects.
#
# Per mechanism, two models (response weights w exactly as in 3c: real = 1,
# synthetic = min(1, real_count / cell size); weights are computed on all
# threads BEFORE any topic filter, so dropping threads leaves the others'
# weights unchanged):
#
#   C. y | weights(w) ~ group + log_length + (1 + synthetic | topic) + (1 | post_id)
#      `group` (real + 9 model x method cells) carries each group's average gap
#      exactly as in 3c's Model A; the random slope on the 0/1 `synthetic`
#      indicator is a topic-specific SHIFT in the gap, shared by all 9
#      synthetic groups, partially pooled across topics. Partial pooling
#      replaces the plan's arbitrary "<5% prevalence" cutoff and any
#      multiplicity correction: small topics are shrunk toward the average,
#      not tested separately. Topic's gap = mean of the 9 group coefficients
#      + that topic's deviation (log-odds scale, exponentiated to ORs).
#
#   D. y | weights(w) ~ group + cancer_relevant + synthetic:cancer_relevant
#                         + log_length + (1 | post_id)
#      A plain fixed interaction for the 518 cancer-relevant threads; no topic.
#
# Outcome: all three mechanisms are binary here. IP and EX: label == 2 (label 1
# never occurs, so this equals label >= 1). ER: label >= 1, i.e. the
# collaborator's own collapse of ER - the 3-level cumulative model would leave
# label-2 cells nearly empty once split by topic (grok has 3 of ~14,300).
#
# Excluded from C only: the two residual categories "Nonmedical" (39 threads)
# and "Medical; other" (361) - not topics, per the collaborator's plan; they
# stay in Models A/B and in D. D drops the 2 threads with no cancer flag.
# Categories under 5% of threads are kept in the model (pooling handles them)
# and flagged `exploratory` in the outputs.
#
# Caveats carried from the collaborator's validation: ~1 label in 4 disagrees
# with the reference standard (mostly LLM-vs-LLM), so moderation estimates are
# attenuated toward zero and a null is weak evidence; weights make the
# likelihood a pseudo-likelihood (posterior widths approximate, see 3c).
#
# Priors: normal(0, 2.5) on fixed effects, student_t(3, 0, 2.5) on the
# intercept and the post_id SD, normal(0, 1) (half, since SDs are positive) on
# the topic intercept/slope SDs - tighter because 15 topics carry little
# information about a variance. adapt_delta = 0.95 for the random-slope fit.
# Sampler, seed, threading: as 3c (threading => the saved fits, not the seed,
# are the reproducible artifact).
#
# Not meant to be run casually: six brms fits on ~137k rows. Run in the
# background: nohup Rscript 3h_topic_moderation_models.R > topic_models.log 2>&1 &

suppressMessages({
  library(dplyr)
  library(brms)
  library(bayestestR)
})

out_dir <- "../../output/5_inference_final"
model_dir <- file.path(out_dir, "models_bayesian")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
topic_path <- "../../input/topics/topic_labels.csv"

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
GROUP_LEVELS <- c("real", as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_")))
RESIDUAL_TOPICS <- c("Nonmedical", "Medical; other")
EXPLORATORY_BELOW <- 0.05   # share of all threads

CHAINS <- 4
THREADS_PER_CHAIN <- 4
ITER <- 2000
WARMUP <- 1000
SEED <- 20260101
ADAPT_DELTA <- 0.95
# Fits that needed more draws to meet max Rhat < 1.01: IP Model C had 1.0106 at
# 2000 iterations (ESS ratio 0.175, no divergences), so it is refit with 4000
# (same warmup). brm()'s `file` cache ignores iter/warmup changes, so the old
# .rds was moved to models_bayesian/superseded_iter2000/ before this run.
ITER_OVERRIDES <- c(topic_IP_C_topic_gap = 4000)
iter_for <- function(cache_name) if (cache_name %in% names(ITER_OVERRIDES)) ITER_OVERRIDES[[cache_name]] else ITER
CREDIBLE_PD <- 0.975
set.seed(SEED)

# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------
cat("Loading data...\n")
results <- read.csv(file.path(out_dir, "inference_results.csv")) %>%
  filter(!is.na(response_post) & response_post != "") %>%
  mutate(
    group = factor(ifelse(source == "real", "real", paste(model, method, sep = "_")), levels = GROUP_LEVELS),
    post_id = factor(post_id),
    synthetic = as.integer(source == "synthetic"),
    log_length = log(pmax(lengths(strsplit(as.character(response_post), "\\s+")), 1)),
    ER_bin = as.integer(ER_label >= 1),
    IP_bin = as.integer(IP_label >= 1),
    EX_bin = as.integer(EX_label >= 1)
  )

real_count <- results %>% filter(source == "real") %>% count(post_id, name = "real_count")
results <- results %>%
  left_join(real_count, by = "post_id") %>%
  mutate(real_count = coalesce(real_count, 0L)) %>%
  group_by(post_id, model, method) %>%
  mutate(cell_n = n()) %>%
  ungroup() %>%
  mutate(w = ifelse(source == "real", 1, pmin(1, real_count / cell_n))) %>%
  filter(w > 0)

# cancer_relevant is exported as the text "True"/"False" (blank when missing);
# map it explicitly rather than trusting read.csv's type guessing.
topics <- read.csv(topic_path, stringsAsFactors = FALSE) %>%
  mutate(cancer_relevant = case_when(tolower(trimws(cancer_relevant)) == "true" ~ 1L,
                                     tolower(trimws(cancer_relevant)) == "false" ~ 0L,
                                     TRUE ~ NA_integer_)) %>%
  select(submission_id, discipline, cancer_relevant)
stopifnot(!anyDuplicated(topics$submission_id),
          sum(topics$cancer_relevant, na.rm = TRUE) == 518,   # documented: 518 cancer-relevant threads
          sum(is.na(topics$cancer_relevant)) == 2)
n_threads_all <- n_distinct(results$submission_id)
results <- results %>% left_join(topics, by = "submission_id")
stopifnot(!anyNA(results$discipline))   # every thread must have a topic label

# short, safe topic codes: brms parameter names can't carry commas/semicolons
topic_lookup <- topics %>%
  filter(submission_id %in% results$submission_id) %>%
  count(discipline, name = "n_threads") %>%
  mutate(share = n_threads / n_threads_all,
         topic = gsub("_+", "_", gsub("[^a-z0-9]+", "_", tolower(discipline))),
         exploratory = share < EXPLORATORY_BELOW,
         residual = discipline %in% RESIDUAL_TOPICS)
results <- results %>% left_join(topic_lookup %>% select(discipline, topic), by = "discipline")

dat_topic <- results %>% filter(!discipline %in% RESIDUAL_TOPICS) %>% mutate(topic = factor(topic))
dat_cancer <- results %>% filter(!is.na(cancer_relevant))

cat(sprintf("all: %d rows / %d threads | topic models: %d rows / %d threads / %d topics | cancer model: %d rows / %d threads\n",
            nrow(results), n_threads_all, nrow(dat_topic), n_distinct(dat_topic$post_id), n_distinct(dat_topic$topic),
            nrow(dat_cancer), n_distinct(dat_cancer$post_id)))

writeLines(c(
  sprintf("date: %s", Sys.time()), sprintf("seed: %d", SEED),
  sprintf("chains: %d, iter: %d, warmup: %d, threads_per_chain: %d, adapt_delta: %s", CHAINS, ITER, WARMUP, THREADS_PER_CHAIN, ADAPT_DELTA),
  sprintf("iteration overrides: %s", paste(names(ITER_OVERRIDES), ITER_OVERRIDES, sep = " = ", collapse = ", ")),
  sprintf("topic file md5: %s", tools::md5sum(topic_path)),
  "weights: synthetic w = min(1, real_count / cell_n), real w = 1",
  sprintf("cmdstan_version: %s", as.character(cmdstanr::cmdstan_version())),
  "", "sessionInfo():", capture.output(sessionInfo())
), file.path(model_dir, "run_metadata_topic.txt"))

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
priors_topic <- c(
  prior_string("normal(0, 2.5)", class = "b"),
  prior_string("student_t(3, 0, 2.5)", class = "Intercept"),
  prior_string("student_t(3, 0, 2.5)", class = "sd", group = "post_id"),
  prior_string("normal(0, 1)", class = "sd", group = "topic")
)
priors_cancer <- c(
  prior_string("normal(0, 2.5)", class = "b"),
  prior_string("student_t(3, 0, 2.5)", class = "Intercept"),
  prior_string("student_t(3, 0, 2.5)", class = "sd")
)

fit_brms <- function(formula, data, cache_name, priors) {
  brm(
    formula = formula, data = data, family = bernoulli("logit"), prior = priors,
    chains = CHAINS, cores = CHAINS,
    threads = if (THREADS_PER_CHAIN > 1) threading(THREADS_PER_CHAIN) else NULL,
    backend = "cmdstanr", iter = iter_for(cache_name), warmup = WARMUP, seed = SEED,
    control = list(adapt_delta = ADAPT_DELTA),
    file = file.path(model_dir, cache_name), file_refit = "on_change"
  )
}

check_convergence_bayes <- function(fit) {
  rhats <- brms::rhat(fit)
  neff <- brms::neff_ratio(fit)
  divergent <- tryCatch(sum(subset(brms::nuts_params(fit), Parameter == "divergent__")$Value), error = function(e) NA_integer_)
  max_rhat <- suppressWarnings(max(rhats, na.rm = TRUE))
  min_neff <- suppressWarnings(min(neff, na.rm = TRUE))
  ok <- is.finite(max_rhat) && max_rhat < 1.01 && !is.na(divergent) && divergent == 0 && min_neff > 0.1
  data.frame(converged = ok, max_rhat = max_rhat, min_neff_ratio = min_neff, divergent_transitions = divergent)
}

# median, 95% HDI, probability of direction for a vector of draws
summ <- function(x) {
  h <- bayestestR::hdi(x, ci = 0.95)
  c(median = median(x), lcl = h$CI_low, ucl = h$CI_high, pd = max(mean(x > 0), mean(x < 0)))
}
summ_or <- function(x, prefix) {
  s <- summ(x)
  setNames(c(exp(s[["median"]]), exp(s[["lcl"]]), exp(s[["ucl"]]), s[["pd"]]),
           paste0(prefix, c("_or", "_lcl", "_ucl", "_pd")))
}

run_mechanism <- function(mech, resp) {
  cat(sprintf("\n=== %s (response %s) ===\n", mech, resp))

  # ---- Model C: topic-varying gap ----
  cat(sprintf("--- %s: Model C (topic moderation, n=%d) ---\n", mech, nrow(dat_topic)))
  fC <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + log_length + (1 + synthetic | topic) + (1 | post_id)", resp)),
                 dat_topic, sprintf("topic_%s_C_topic_gap", mech), priors_topic)
  convC <- check_convergence_bayes(fC)
  cat(sprintf("  converged: %s (max Rhat %.4f, min ESS ratio %.3f, divergences %s)\n",
              convC$converged, convC$max_rhat, convC$min_neff_ratio, convC$divergent_transitions))

  dr <- as.data.frame(brms::as_draws_df(fC))
  b_syn <- dr[, grep("^b_group", names(dr)), drop = FALSE]
  stopifnot(ncol(b_syn) == 9)
  avg_gap <- rowMeans(b_syn)                       # average synthetic-vs-real gap (log-odds)
  re_cols <- grep("^r_topic\\[.*,synthetic\\]$", names(dr), value = TRUE)
  stopifnot(length(re_cols) == n_distinct(dat_topic$topic))

  by_topic <- bind_rows(lapply(re_cols, function(cn) {
    t <- sub("^r_topic\\[(.*),synthetic\\]$", "\\1", cn)
    dev <- dr[[cn]]
    c(topic = t, as.list(summ_or(avg_gap + dev, "gap")), as.list(summ_or(dev, "shift")))
  })) %>%
    mutate(across(-topic, as.numeric), mechanism = mech) %>%
    left_join(topic_lookup %>% select(topic, discipline, n_threads, share, exploratory), by = "topic") %>%
    mutate(shift_credible = shift_pd > CREDIBLE_PD & (shift_lcl > 1 | shift_ucl < 1))
  overall <- data.frame(topic = "(average over topics)", discipline = "(average over topics)",
                        t(summ_or(avg_gap, "gap")), mechanism = mech)

  sd_draws <- list(topic_intercept_sd = dr$sd_topic__Intercept, topic_slope_sd = dr$sd_topic__synthetic,
                   post_id_sd = dr$sd_post_id__Intercept)
  sd_tbl <- bind_rows(lapply(names(sd_draws), function(nm) {
    h <- bayestestR::hdi(sd_draws[[nm]], ci = 0.95)
    data.frame(mechanism = mech, parameter = nm, median = median(sd_draws[[nm]]), lcl = h$CI_low, ucl = h$CI_high)
  }))

  # ---- Model D: cancer relevance ----
  cat(sprintf("--- %s: Model D (cancer relevance, n=%d) ---\n", mech, nrow(dat_cancer)))
  fD <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + cancer_relevant + synthetic:cancer_relevant + log_length + (1 | post_id)", resp)),
                 dat_cancer, sprintf("topic_%s_D_cancer", mech), priors_cancer)
  convD <- check_convergence_bayes(fD)
  cat(sprintf("  converged: %s (max Rhat %.4f, min ESS ratio %.3f, divergences %s)\n",
              convD$converged, convD$max_rhat, convD$min_neff_ratio, convD$divergent_transitions))
  dd <- as.data.frame(brms::as_draws_df(fD))
  gap0 <- rowMeans(dd[, grep("^b_group", names(dd)), drop = FALSE])
  int_col <- grep("^b_synthetic:cancer_relevant$|^b_cancer_relevant:synthetic$", names(dd), value = TRUE)
  stopifnot(length(int_col) == 1)
  cancer_tbl <- data.frame(mechanism = mech, t(c(
    summ_or(gap0, "gap_not_cancer"), summ_or(gap0 + dd[[int_col]], "gap_cancer"),
    summ_or(dd[[int_col]], "shift"), summ_or(dd$b_cancer_relevant, "cancer_main")
  )))

  list(
    convergence = bind_rows(cbind(mechanism = mech, model = "C_topic_gap", convC),
                            cbind(mechanism = mech, model = "D_cancer", convD)),
    by_topic = bind_rows(by_topic, overall), sd = sd_tbl, cancer = cancer_tbl
  )
}

res <- list(
  ER = run_mechanism("ER", "ER_bin"),
  IP = run_mechanism("IP", "IP_bin"),
  EX = run_mechanism("EX", "EX_bin")
)

write.csv(bind_rows(lapply(res, `[[`, "convergence")), file.path(out_dir, "topic_convergence.csv"), row.names = FALSE)
write.csv(bind_rows(lapply(res, `[[`, "by_topic")), file.path(out_dir, "topic_gap_by_topic.csv"), row.names = FALSE)
write.csv(bind_rows(lapply(res, `[[`, "sd")), file.path(out_dir, "topic_sd_summary.csv"), row.names = FALSE)
write.csv(bind_rows(lapply(res, `[[`, "cancer")), file.path(out_dir, "topic_cancer_moderation.csv"), row.names = FALSE)
write.csv(topic_lookup, file.path(out_dir, "topic_lookup.csv"), row.names = FALSE)

cat("\n=== Convergence ===\n"); print(bind_rows(lapply(res, `[[`, "convergence")))
cat("\n=== SD of topic effects ===\n"); print(bind_rows(lapply(res, `[[`, "sd")))
cat("\n=== Cancer relevance ===\n"); print(bind_rows(lapply(res, `[[`, "cancer")))
cat("\nWrote topic_{convergence,gap_by_topic,sd_summary,cancer_moderation,lookup}.csv to", out_dir, "\n")
