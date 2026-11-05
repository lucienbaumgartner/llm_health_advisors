#!/usr/bin/env Rscript
#
# Prompting-strategy analyses (follow-up to 3i; nothing from 3c/3h/3i is read or
# overwritten). Data preparation, covariates, topics, weights, priors, sampler,
# seed and convergence rule are exactly those of 3i_topic_llm_moderation_v2.R.
#
# Per mechanism (ER, IP, EX), fitted on the same rows as 3i's topic models:
#
#  L0 / L1. Strategy main effects within each AI model, without / with the
#       log-length adjustment:
#         y | weights(w) ~ group + COV [+ log_length] + (1 | post_id)
#       The nine group coefficients give each model-by-strategy gap versus the real
#       comments and the strategy differences within a model. Fitting both lets the
#       report show the gap before and after length adjustment side by side.
#
#  H.   Severity moderation per model-by-strategy group (nine slopes instead of 3i's
#       three per-model slopes):
#         y | weights(w) ~ group + COV + ig_<g>:sev_c (9 terms) + synthetic:sev_na
#                            + log_length + (1 + synthetic | topic) + (1 | post_id)
#       ig_<g> are 0/1 indicators of group g (real = all 0). Slope g = extra change in
#       group g's gap per severity level. Summarised per group, per model (mean of its
#       three strategies), per strategy (mean over the three models) and as
#       strategy differences within a model.
#
#  I.   Does topic moderation vary by model or strategy at all? Topic shifts of the
#       synthetic-vs-real gap with a common shift per topic plus three variance
#       components, each with ONE shared SD:
#         (1 + synthetic | topic)            common topic shift (as 3i Model C2)
#         (0 + synthetic | topic:model)      extra shift per topic x AI model
#         (0 + synthetic | topic:strategy)   extra shift per topic x prompting strategy
#         (0 + synthetic | topic:model:strategy)  extra shift per topic x group
#       Real comments (synthetic = 0) are assigned a dummy level, which carries no data.
#       The SDs say how much topic moderation varies by model / strategy / the nine
#       groups; the near-zero case is the answer "not at all". Full 9-group random
#       slopes with correlations (55 covariance parameters from 11 topics) are not
#       fitted.
#
# Then, for IP only (STAGES includes "sens"): the pooled topic model of 3i (C2)
# without the random topic intercept, whose SD is near 0 and probably causes the
# divergent transitions (adapt_delta 0.99); compared with the 3i shifts.
#
# Outputs are rewritten after each mechanism finishes (v3_*.csv, run_metadata_v3.txt),
# so partial results survive an interruption. Fits are cached (v3_* in models_bayesian/).
#
# Run:  nohup Rscript 3j_strategy_models.R > strategy_v3.log 2>&1 &
# Subset: MECHS=IP Rscript 3j_strategy_models.R;  STAGES=main|sens|main,sens (default both)
# Quick check (small subsample, few iterations, separate cache and test_ outputs):
#         TEST_FRAC=0.05 Rscript 3j_strategy_models.R
#
# Subsampling sensitivity check (SUBS_SEED=<integer>): instead of weighting, each
# synthetic cell (thread x model x method) is randomly thinned to the number of real
# comments of that thread (cells with fewer comments are kept whole), all weights are
# 1, and everything else is as above. Cache and outputs are prefixed sub<seed>_v3_
# resp. v3sub<seed>_; FITS selects which fits to run (default L0,L1,H,I).
#         SUBS_SEED=1 nohup Rscript 3j_strategy_models.R > strategy_v3_sub1.log 2>&1 &
#         SUBS_SEED=2 FITS=L1 STAGES=main Rscript 3j_strategy_models.R

suppressMessages({
  library(dplyr)
  library(brms)
  library(bayestestR)
})

out_dir <- "../../output/5_inference_final"
topic_path <- "../../input/topics/topic_labels.csv"
cov_dir <- "../../input/covariates"
MECHS <- strsplit(Sys.getenv("MECHS", "ER,IP,EX"), ",")[[1]]
STAGES <- strsplit(Sys.getenv("STAGES", "main,sens"), ",")[[1]]
TEST_FRAC <- as.numeric(Sys.getenv("TEST_FRAC", "0"))
TEST <- TEST_FRAC > 0
model_dir <- file.path(out_dir, "models_bayesian", if (TEST) "test_v3" else ".")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
SUBS_SEED <- as.integer(Sys.getenv("SUBS_SEED", "0"))
SUBS <- SUBS_SEED > 0
FITS <- strsplit(Sys.getenv("FITS", "L0,L1,H,I"), ",")[[1]]
if (SUBS) STAGES <- setdiff(STAGES, "sens")   # the IP sensitivity fit is not part of the subsampling check
OUT_PREFIX <- if (TEST) "test_v3_" else if (SUBS) sprintf("v3sub%d_%s_", SUBS_SEED, paste(MECHS, collapse = "")) else "v3_"
CACHE_PREFIX <- if (SUBS) sprintf("sub%d_v3_", SUBS_SEED) else "v3_"

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
GROUP_LEVELS <- c("real", as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_")))
SYN_GROUPS <- GROUP_LEVELS[-1]
RESIDUAL_TOPICS <- c("Nonmedical", "Medical; other")
EXPLORATORY_BELOW <- 0.05

TOPIC_MERGE <- c(
  "Pulmonology" = "Cardiopulmonary conditions",
  "Cardiovascular" = "Cardiopulmonary conditions",
  "Dentistry" = "Oral and otolaryngologic conditions",
  "Otolaryngology" = "Oral and otolaryngologic conditions",
  "Ophthalmology" = "Neurologic and visual conditions",
  "Neurology" = "Neurologic and visual conditions",
  "Hematology and immunology" = "Hematologic, immunologic, and infectious conditions",
  "Infectious diseases" = "Hematologic, immunologic, and infectious conditions"
)
merge_topic <- function(x) ifelse(x %in% names(TOPIC_MERGE), unname(TOPIC_MERGE[x]), x)
topic_code <- function(x) gsub("_+", "_", gsub("[^a-z0-9]+", "_", tolower(x)))

COV <- "subject + age10 + age_na + sex3 + sev_c + sev_na + real_match + real_mismatch"

CHAINS <- 4
THREADS_PER_CHAIN <- 4
ITER <- if (TEST) 300 else 2000
WARMUP <- if (TEST) 150 else 1000
SEED <- 20260101
ADAPT_DELTA <- 0.95
ITER_OVERRIDES <- c(v3_IP_H_severity_by_group = 4000, v3_EX_H_severity_by_group = 4000, v3_EX_I_topic_by_group = 4000, v3_IP_C2_nointercept = 4000)   # e.g. c(v3_IP_I_topic_by_group = 4000); move the old .rds aside first (brm's file cache ignores iter changes)
iter_for <- function(cache_name) if (cache_name %in% names(ITER_OVERRIDES)) ITER_OVERRIDES[[cache_name]] else ITER
CREDIBLE_PD <- 0.975
set.seed(SEED)

# ---------------------------------------------------------------------
# data (identical to 3i, plus group indicators and topic-by-group labels)
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

if (SUBS) {
  # thin every synthetic cell to the thread's number of real comments; no weights
  set.seed(SUBS_SEED)
  n_before <- sum(results$source == "synthetic")
  results <- results %>%
    group_by(post_id, model, method) %>%
    mutate(keep = source == "real" | cell_n <= real_count | seq_len(n()) %in% sample.int(n(), min(first(real_count), n()))) %>%
    ungroup() %>%
    filter(keep) %>%
    mutate(w = 1) %>%
    select(-keep)
  cat(sprintf("SUBSAMPLING (seed %d): synthetic comments %d -> %d, real %d\n", SUBS_SEED, n_before,
              sum(results$source == "synthetic"), sum(results$source == "real")))
  set.seed(SEED)
}

topics <- read.csv(topic_path, stringsAsFactors = FALSE) %>%
  select(submission_id, discipline)
demo <- read.csv(file.path(cov_dir, "poster_demographics.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, subject, age_years, sex)
sev <- read.csv(file.path(cov_dir, "severity.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, severity_score)
roles <- read.csv(file.path(cov_dir, "first_responder_roles.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, specialty_kind, specialty_discipline)
stopifnot(!anyDuplicated(topics$submission_id), !anyDuplicated(demo$submission_id),
          !anyDuplicated(sev$submission_id), !anyDuplicated(roles$submission_id))

threads <- topics %>%
  left_join(demo, by = "submission_id") %>%
  left_join(sev, by = "submission_id") %>%
  left_join(roles, by = "submission_id") %>%
  mutate(
    topic_label = merge_topic(discipline),
    spec_merged = ifelse(specialty_kind %in% "body-system", merge_topic(specialty_discipline), NA_character_),
    match3 = case_when(is.na(spec_merged) ~ "undeclared",
                       spec_merged == topic_label ~ "match",
                       TRUE ~ "mismatch"),
    subject = factor(ifelse(subject %in% c("first_person", "third_party"), subject, "unknown"),
                     levels = c("first_person", "third_party", "unknown")),
    sex3 = factor(case_when(sex %in% "female" ~ "female", sex %in% "male" ~ "male", TRUE ~ "unknown"),
                  levels = c("female", "male", "unknown")),
    age_na = as.integer(is.na(age_years)),
    age10 = ifelse(is.na(age_years), 0, (age_years - median(age_years, na.rm = TRUE)) / 10),
    sev_na = as.integer(is.na(severity_score)),
    sev_c = ifelse(is.na(severity_score), 0, severity_score - mean(severity_score, na.rm = TRUE))
  )

results <- results %>%
  left_join(threads %>% select(submission_id, discipline, topic_label, match3,
                               subject, sex3, age10, age_na, sev_c, sev_na),
            by = "submission_id") %>%
  mutate(real_match = as.integer(source == "real" & match3 == "match"),
         real_mismatch = as.integer(source == "real" & match3 == "mismatch"),
         topic = topic_code(topic_label),
         model_c = sub("_.*$", "", as.character(group)),
         method_c = sub("^.*_", "", as.character(group)),
         # real comments get a dummy level: their slope on `synthetic` is 0, so it carries no data
         tm = ifelse(synthetic == 1, paste(topic, model_c, sep = "__"), "none"),
         tt = ifelse(synthetic == 1, paste(topic, method_c, sep = "__"), "none"),
         tmm = ifelse(synthetic == 1, paste(topic, model_c, method_c, sep = "__"), "none"))
for (g in SYN_GROUPS) results[[paste0("ig_", g)]] <- as.integer(results$group == g)
stopifnot(!anyNA(results$discipline),
          !anyNA(results[, c("subject", "sex3", "age10", "age_na", "sev_c", "sev_na", "real_match", "real_mismatch")]))

n_threads_all <- n_distinct(results$submission_id)
topic_lookup <- results %>%
  distinct(submission_id, topic, topic_label, discipline) %>%
  filter(!discipline %in% RESIDUAL_TOPICS) %>%
  count(topic, topic_label, name = "n_threads") %>%
  mutate(share = n_threads / n_threads_all, exploratory = share < EXPLORATORY_BELOW)

dat_topic <- results %>% filter(!discipline %in% RESIDUAL_TOPICS)
if (TEST) {
  keep <- sample(unique(dat_topic$post_id), ceiling(TEST_FRAC * n_distinct(dat_topic$post_id)))
  dat_topic <- dat_topic %>% filter(post_id %in% keep)
  cat(sprintf("TEST MODE: %d%% of threads, iter %d\n", round(100 * TEST_FRAC), ITER))
}
dat_topic <- dat_topic %>% mutate(topic = factor(topic), tm = factor(tm), tt = factor(tt), tmm = factor(tmm))
n_topics <- n_distinct(dat_topic$topic)
stopifnot(n_topics == 11, n_distinct(dat_topic$tm) == 34, n_distinct(dat_topic$tt) == 34, n_distinct(dat_topic$tmm) == 100)

cat(sprintf("topic-model data: %d rows / %d threads / %d topics\n", nrow(dat_topic), n_distinct(dat_topic$post_id), n_topics))

md5 <- function(f) unname(tools::md5sum(f))
writeLines(c(
  sprintf("date: %s", Sys.time()), sprintf("seed: %d", SEED), sprintf("test mode: %s", TEST),
  sprintf("chains: %d, iter: %d, warmup: %d, threads_per_chain: %d, adapt_delta: %s", CHAINS, ITER, WARMUP, THREADS_PER_CHAIN, ADAPT_DELTA),
  sprintf("iteration overrides: %s", if (length(ITER_OVERRIDES)) paste(names(ITER_OVERRIDES), ITER_OVERRIDES, sep = " = ", collapse = ", ") else "none"),
  sprintf("topic file md5: %s", md5(topic_path)),
  sprintf("covariate md5: demographics %s, severity %s, roles %s",
          md5(file.path(cov_dir, "poster_demographics.csv")), md5(file.path(cov_dir, "severity.csv")),
          md5(file.path(cov_dir, "first_responder_roles.csv"))),
  sprintf("covariate terms: %s", COV),
  if (SUBS) sprintf("SUBSAMPLING seed %d: synthetic cells thinned to the thread's real_count, all w = 1; fits: %s", SUBS_SEED, paste(FITS, collapse = ","))
  else "weights: synthetic w = min(1, real_count / cell_n), real w = 1",
  sprintf("cmdstan_version: %s", as.character(cmdstanr::cmdstan_version())),
  "", "sessionInfo():", capture.output(sessionInfo())
), file.path(model_dir, paste0("run_metadata_", sub("_$", "", OUT_PREFIX), ".txt")))

# ---------------------------------------------------------------------
# helpers (as 3i)
# ---------------------------------------------------------------------
priors_common <- c(
  prior_string("normal(0, 2.5)", class = "b"),
  prior_string("student_t(3, 0, 2.5)", class = "Intercept"),
  prior_string("student_t(3, 0, 2.5)", class = "sd", group = "post_id")
)
priors_topic <- c(priors_common, prior_string("normal(0, 1)", class = "sd", group = "topic"))
priors_byg <- c(priors_topic,
                prior_string("normal(0, 1)", class = "sd", group = "tm"),
                prior_string("normal(0, 1)", class = "sd", group = "tt"),
                prior_string("normal(0, 1)", class = "sd", group = "tmm"))

fit_brms <- function(formula, data, cache_name, priors, adapt_delta = ADAPT_DELTA) {
  cache_name <- sub("^v3_", CACHE_PREFIX, cache_name)
  brm(
    formula = formula, data = data, family = bernoulli("logit"), prior = priors,
    chains = CHAINS, cores = CHAINS,
    threads = if (THREADS_PER_CHAIN > 1) threading(THREADS_PER_CHAIN) else NULL,
    backend = "cmdstanr", iter = iter_for(sub(CACHE_PREFIX, "v3_", cache_name, fixed = TRUE)), warmup = WARMUP, seed = SEED,
    control = list(adapt_delta = adapt_delta),
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
report_conv <- function(label, cv) {
  cat(sprintf("  %s converged: %s (max Rhat %.4f, min ESS ratio %.3f, divergences %s)\n",
              label, cv$converged, cv$max_rhat, cv$min_neff_ratio, cv$divergent_transitions))
}

summ <- function(x) {
  h <- bayestestR::hdi(x, ci = 0.95)
  c(median = median(x), lcl = h$CI_low, ucl = h$CI_high, pd = max(mean(x > 0), mean(x < 0)))
}
effect_row <- function(x, ...) {
  s <- summ(x)
  data.frame(..., or = exp(s[["median"]]), lcl = exp(s[["lcl"]]), ucl = exp(s[["ucl"]]), pd = s[["pd"]])
}
sd_row <- function(x, mech, parameter, model) {
  h <- bayestestR::hdi(x, ci = 0.95)
  data.frame(mechanism = mech, model = model, parameter = parameter, median = median(x), lcl = h$CI_low, ucl = h$CI_high)
}

# Per-group coefficients gd (named list over the nine synthetic groups; log-odds draws)
# -> the value for each group, each AI model (mean of its three strategies), each
# strategy (mean over the three models), and the strategy differences. For the
# group coefficients of L0/L1 the value is the gap versus real; for H it is the
# extra change in the gap per severity level.
group_effects <- function(gd, spec, mech, quantity) {
  stopifnot(setequal(names(gd), SYN_GROUPS))
  mod_mean <- lapply(setNames(MODEL_LEVELS, MODEL_LEVELS), function(L) Reduce(`+`, gd[paste(L, METHOD_LEVELS, sep = "_")]) / 3)
  str_mean <- lapply(setNames(METHOD_LEVELS, METHOD_LEVELS), function(m) Reduce(`+`, gd[paste(MODEL_LEVELS, m, sep = "_")]) / 3)
  pairs <- list(c("dg", "sag"), c("dg", "sagii"), c("sag", "sagii"))
  rows <- c(
    lapply(SYN_GROUPS, function(g) effect_row(gd[[g]], spec = spec, mechanism = mech, quantity = quantity, level = "group", who = g, contrast = "")),
    lapply(MODEL_LEVELS, function(L) effect_row(mod_mean[[L]], spec = spec, mechanism = mech, quantity = quantity, level = "model", who = L, contrast = "")),
    lapply(METHOD_LEVELS, function(m) effect_row(str_mean[[m]], spec = spec, mechanism = mech, quantity = quantity, level = "strategy", who = m, contrast = "")),
    unlist(lapply(MODEL_LEVELS, function(L) lapply(pairs, function(p)
      effect_row(gd[[paste(L, p[1], sep = "_")]] - gd[[paste(L, p[2], sep = "_")]], spec = spec, mechanism = mech, quantity = quantity,
                 level = "strategy_within_model", who = L, contrast = paste(p, collapse = " - ")))), recursive = FALSE),
    lapply(pairs, function(p) effect_row(str_mean[[p[1]]] - str_mean[[p[2]]], spec = spec, mechanism = mech, quantity = quantity,
                                         level = "strategy_across_models", who = "all", contrast = paste(p, collapse = " - ")))
  )
  bind_rows(rows)
}

# deviations of one random-slope term on `synthetic` ("tm", "tt" or "tmm"), parsed to topic / model / strategy
deviation_table <- function(dr, term, mech) {
  pat <- sprintf("^r_%s\\[(.*),synthetic\\]$", term)
  cols <- grep(pat, names(dr), value = TRUE)
  bind_rows(lapply(cols, function(cn) {
    lev <- sub(pat, "\\1", cn)
    if (lev == "none") return(NULL)
    parts <- strsplit(lev, "__", fixed = TRUE)[[1]]
    effect_row(dr[[cn]], mechanism = mech, term = term, topic = parts[1],
               model = if (term %in% c("tm", "tmm")) parts[2] else NA_character_,
               strategy = if (term == "tt") parts[2] else if (term == "tmm") parts[3] else NA_character_)
  })) %>%
    left_join(topic_lookup, by = "topic") %>%
    mutate(credible = pd > CREDIBLE_PD & (lcl > 1 | ucl < 1))
}

run_mechanism <- function(mech) {
  resp <- paste0(mech, "_bin")
  cat(sprintf("\n=== %s (response %s) ===\n", mech, resp))
  f <- function(rhs) as.formula(sprintf("%s | weights(w) ~ %s", resp, rhs))
  bg <- function(dr) setNames(lapply(SYN_GROUPS, function(g) dr[[paste0("b_group", g)]]), SYN_GROUPS)

  # ---- L0 / L1: strategy main effects, without / with the length adjustment ----
  convL0 <- convL1 <- convH <- convI <- NULL
  eff_L0 <- eff_L1 <- eff_H <- len_tbl <- sd_I <- dev_I <- NULL
  if ("L0" %in% FITS) {
  cat(sprintf("--- %s: L0 (no length adjustment, n=%d) ---\n", mech, nrow(dat_topic)))
  fL0 <- fit_brms(f(sprintf("group + %s + (1 | post_id)", COV)), dat_topic, sprintf("v3_%s_L0_nolength", mech), priors_common)
  convL0 <- check_convergence_bayes(fL0); report_conv("L0", convL0)
  dL0 <- as.data.frame(brms::as_draws_df(fL0))
  eff_L0 <- group_effects(bg(dL0), "no length adjustment", mech, "gap vs real")
  rm(fL0)
  }

  if ("L1" %in% FITS) {
  cat(sprintf("--- %s: L1 (length adjusted, n=%d) ---\n", mech, nrow(dat_topic)))
  fL1 <- fit_brms(f(sprintf("group + %s + log_length + (1 | post_id)", COV)), dat_topic, sprintf("v3_%s_L1_length", mech), priors_common)
  convL1 <- check_convergence_bayes(fL1); report_conv("L1", convL1)
  dL1 <- as.data.frame(brms::as_draws_df(fL1))
  eff_L1 <- group_effects(bg(dL1), "length adjusted", mech, "gap vs real")
  len_tbl <- effect_row(dL1$b_log_length, mechanism = mech, term = "log_length (per log word)")
  rm(fL1)
  }

  # ---- H: severity moderation per model-by-strategy group ----
  if ("H" %in% FITS) {
  cat(sprintf("--- %s: H (severity per group, n=%d) ---\n", mech, nrow(dat_topic)))
  sev_terms <- paste0("ig_", SYN_GROUPS, ":sev_c", collapse = " + ")
  fH <- fit_brms(f(sprintf("group + %s + %s + synthetic:sev_na + log_length + (1 + synthetic | topic) + (1 | post_id)", COV, sev_terms)),
                 dat_topic, sprintf("v3_%s_H_severity_by_group", mech), priors_topic)
  convH <- check_convergence_bayes(fH); report_conv("H", convH)
  dH <- as.data.frame(brms::as_draws_df(fH))
  slope_col <- function(g) {
    cols <- grep(sprintf("^b_(ig_%s:sev_c|sev_c:ig_%s)$", g, g), names(dH), value = TRUE)
    stopifnot(length(cols) == 1); cols
  }
  eff_H <- group_effects(setNames(lapply(SYN_GROUPS, function(g) dH[[slope_col(g)]]), SYN_GROUPS),
                         "severity by group", mech, "extra change in gap per severity level")
  rm(fH)
  }

  if ("I" %in% FITS) {
  # ---- I: topic moderation varying by model / strategy / group ----
  cat(sprintf("--- %s: I (topic x model/strategy variance components, n=%d) ---\n", mech, nrow(dat_topic)))
  fI <- fit_brms(f(sprintf("group + %s + log_length + (1 + synthetic | topic) + (0 + synthetic | tm) + (0 + synthetic | tt) + (0 + synthetic | tmm) + (1 | post_id)", COV)),
                 dat_topic, sprintf("v3_%s_I_topic_by_group", mech), priors_byg)
  convI <- check_convergence_bayes(fI); report_conv("I", convI)
  dI <- as.data.frame(brms::as_draws_df(fI))
  sd_I <- bind_rows(
    sd_row(dI$sd_topic__Intercept, mech, "topic_intercept_sd", "I"),
    sd_row(dI$sd_topic__synthetic, mech, "topic_slope_sd (common shift)", "I"),
    sd_row(dI$sd_tm__synthetic, mech, "topic_x_model_sd", "I"),
    sd_row(dI$sd_tt__synthetic, mech, "topic_x_strategy_sd", "I"),
    sd_row(dI$sd_tmm__synthetic, mech, "topic_x_model_x_strategy_sd", "I"),
    sd_row(dI$sd_post_id__Intercept, mech, "post_id_sd", "I"))
  dev_I <- bind_rows(deviation_table(dI, "tm", mech), deviation_table(dI, "tt", mech), deviation_table(dI, "tmm", mech))
  rm(fI)
  }

  list(
    convergence = bind_rows(
      if (!is.null(convL0)) cbind(mechanism = mech, model = "L0_nolength", convL0),
      if (!is.null(convL1)) cbind(mechanism = mech, model = "L1_length", convL1),
      if (!is.null(convH)) cbind(mechanism = mech, model = "H_severity_by_group", convH),
      if (!is.null(convI)) cbind(mechanism = mech, model = "I_topic_by_group", convI)),
    group_effects = bind_rows(eff_L0, eff_L1, eff_H),
    length_effect = len_tbl, sd = sd_I, deviations = dev_I
  )
}

write_all <- function(res) {
  pick <- function(nm) bind_rows(lapply(res, `[[`, nm))
  write.csv(pick("convergence"), file.path(out_dir, paste0(OUT_PREFIX, "convergence.csv")), row.names = FALSE)
  write.csv(pick("group_effects"), file.path(out_dir, paste0(OUT_PREFIX, "group_effects.csv")), row.names = FALSE)
  write.csv(pick("length_effect"), file.path(out_dir, paste0(OUT_PREFIX, "length_effect.csv")), row.names = FALSE)
  write.csv(pick("sd"), file.path(out_dir, paste0(OUT_PREFIX, "topic_by_group_sd.csv")), row.names = FALSE)
  write.csv(pick("deviations"), file.path(out_dir, paste0(OUT_PREFIX, "topic_by_group_deviations.csv")), row.names = FALSE)
  cat(sprintf("Wrote %s*.csv (mechanisms done: %s)\n", OUT_PREFIX, paste(names(res), collapse = ", ")))
}

res <- list()
if ("main" %in% STAGES) {
  for (m in MECHS) {
    res[[m]] <- run_mechanism(m)
    write_all(res)
  }
  cat("\n=== Convergence ===\n"); print(bind_rows(lapply(res, `[[`, "convergence")))
  cat("\n=== SDs of the topic variance components ===\n"); print(bind_rows(lapply(res, `[[`, "sd")))
}

# ---------------------------------------------------------------------
# IP sensitivity: pooled topic model without the random topic intercept
# ---------------------------------------------------------------------
if ("sens" %in% STAGES && "IP" %in% MECHS) {
  cat("\n=== IP sensitivity: pooled topic model without random topic intercept (adapt_delta 0.99) ===\n")
  fS <- fit_brms(as.formula(sprintf("IP_bin | weights(w) ~ group + %s + log_length + (0 + synthetic | topic) + (1 | post_id)", COV)),
                 dat_topic, "v3_IP_C2_nointercept", priors_topic, adapt_delta = 0.99)
  convS <- check_convergence_bayes(fS); report_conv("C2 without topic intercept", convS)
  dS <- as.data.frame(brms::as_draws_df(fS))
  avg_gap <- rowMeans(dS[, grep("^b_group", names(dS)), drop = FALSE])
  pat <- "^r_topic\\[(.*),synthetic\\]$"
  cols <- grep(pat, names(dS), value = TRUE); stopifnot(length(cols) == n_topics)
  sens <- bind_rows(lapply(cols, function(cn) {
    cbind(effect_row(dS[[cn]], topic = sub(pat, "\\1", cn), quantity = "shift"),
          gap_or = exp(median(avg_gap + dS[[cn]])))
  })) %>% left_join(topic_lookup, by = "topic") %>%
    mutate(credible = pd > CREDIBLE_PD & (lcl > 1 | ucl < 1))
  v2_path <- file.path(out_dir, "v2_topic_gap_by_topic.csv")
  if (file.exists(v2_path)) {
    v2 <- read.csv(v2_path) %>% filter(mechanism == "IP", !is.na(topic)) %>% select(topic, v2_shift_or = shift_or, v2_shift_pd = shift_pd, v2_gap_or = gap_or)
    sens <- left_join(sens, v2, by = "topic")
  }
  write.csv(cbind(mechanism = "IP", sens), file.path(out_dir, paste0(OUT_PREFIX, "IP_sensitivity_no_topic_intercept.csv")), row.names = FALSE)
  write.csv(cbind(mechanism = "IP", model = "C2_no_topic_intercept", convS),
            file.path(out_dir, paste0(OUT_PREFIX, "IP_sensitivity_convergence.csv")), row.names = FALSE)
  print(sens %>% select(topic, or, lcl, ucl, pd, credible, any_of(c("v2_shift_or", "v2_shift_pd"))))
  cat("Wrote IP sensitivity outputs\n")
}
cat("\nDone.\n")
