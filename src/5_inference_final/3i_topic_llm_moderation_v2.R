#!/usr/bin/env Rscript
#
# v2 of the topic-moderation analysis (3h is left untouched; its fits and
# outputs are what the first topic report was built on). Three changes:
#
#  1. Merged topics (13 -> 11 substantive topics):
#       Pulmonology + Cardiovascular            -> Cardiopulmonary conditions
#       Dentistry + Otolaryngology              -> Oral and otolaryngologic conditions
#       Ophthalmology + Neurology               -> Neurologic and visual conditions
#       Hematology and immunology + Infectious diseases
#                                               -> Hematologic, immunologic, and infectious conditions
#     "Nonmedical" and "Medical; other" stay excluded from the topic models
#     (residual categories); everything else is unchanged.
#
#  2. LLM-specific moderation (Model E): does the topic shift the real-vs-AI
#     gap differently for Gemini, OpenAI and Grok? (3h pooled all nine
#     model x method groups into one shift per topic.)
#
#  3. The coauthor's covariates (src/Covariates in cmstewart/llm_health_advisors),
#     added as main effects to every model:
#       subject        first_person / third_party / unknown (who the post is about)
#       age10, age_na  age in decades, centred at the observed median; missing
#                      (11.8%) set to 0 with an indicator
#       sex3           female / male / unknown (missing 17%)
#       sev_c, sev_na  triage severity 1-4 centred at the observed mean;
#                      "not_applicable" and the 1 blank are NOT scored as the
#                      low end, they get 0 plus an indicator (per the README)
#       real_match,    first responder's declared body-system specialty matches /
#       real_mismatch  does not match the thread topic, versus no body-system
#                      specialty declared (reference, ~83% of threads). Entered
#                      for REAL comments only (0 for synthetic): it describes
#                      who answered, which the AI comments know nothing about
#                      (coding confirmed with the project lead). Match is
#                      recomputed on the MERGED topics, so e.g. a pulmonologist
#                      answering a cardiovascular post is a match.
#     Not used, per the coauthor's READMEs: weight/height/BMI (missingness
#     tracks topic; BMI drops ~3/4 of threads), race (0.7%), smoking/drinking
#     (11%/5%), hours_to_first_response (rejected as a severity measure).
#
#     NOTE on what these covariates can do: all are thread-level, and every
#     model already has a thread random intercept (post_id), so as MAIN
#     effects they only explain between-thread baseline variation (this lets
#     them affect real and AI comments alike); they do not change the
#     within-thread real-vs-AI contrast. A covariate changes the GAP only as a
#     moderator (covariate x synthetic). Severity, the one the coauthor's README
#     proposes as a moderator, is tested that way in Model F below.
#
# Models per mechanism (binary outcomes and response weights exactly as in 3h):
#   C2. y | weights(w) ~ group + COV + log_length + (1 + synthetic | topic) + (1 | post_id)
#       topic shift in the pooled real-vs-AI gap (as 3h Model C, new topics + COV)
#   D2. y | weights(w) ~ group + cancer_relevant + synthetic:cancer_relevant + COV
#                          + log_length + (1 | post_id)
#   E.  y | weights(w) ~ group + COV + log_length + (1 + gem + opn + grk | topic) + (1 | post_id)
#       gem/opn/grk are 0/1 indicators for that LLM's comments (real = all 0); each
#       topic gets its own shift for each LLM, with the three shifts allowed to
#       correlate (LKJ(2) prior: shrinks the correlations moderately toward 0,
#       since 11 topics cannot pin down six correlations). An LLM's gap in a
#       topic = mean of its three (method) group coefficients + the topic's
#       shift for that LLM. The shift is shared across an LLM's three methods.
#
#   F.  y | weights(w) ~ group + COV + gem:sev_c + opn:sev_c + grk:sev_c + synthetic:sev_na
#                          + log_length + (1 + synthetic | topic) + (1 | post_id)
#       Severity as a moderator, separately for each LLM. sev_c (severity 1-4,
#       centred) is also in COV, so severity can raise or lower empathy in real
#       AND AI comments alike; the gem/opn/grk x sev_c terms are the EXTRA change
#       in that LLM's gap per severity level. exp(slope) < 1: the AI comments fall
#       further behind the real ones as severity rises; > 1: they catch up. The
#       mean of the three slopes is reported as the pooled moderation. Threads
#       with no usable severity (sev_na = 1, sev_c = 0) contribute nothing to the
#       slopes; synthetic:sev_na just lets their overall gap differ. Topic random
#       effects stay in so that severity is not standing in for topic (severity
#       does vary by topic).
#
# Priors: as 3h (normal(0, 2.5) on fixed effects incl. covariates, which are
# on interpretable scales; student_t(3, 0, 2.5) intercept and thread SD;
# normal(0, 1) on topic SDs). Same sampler, seed, threading, weights, and
# convergence rule as 3h/3c. Reproducibility rests on the saved fits.
#
# New cache names (v2_*) and outputs (v2_*.csv, run_metadata_v2.txt): nothing
# from 3h/3c is read or overwritten.
#
# NOT RUN YET. Twelve brms fits on ~137k rows; E (4 random effects per topic)
# will be the slowest. Run in the background:
#   nohup Rscript 3i_topic_llm_moderation_v2.R > topic_v2.log 2>&1 &
# Subset: MECHS=IP,EX Rscript 3i_topic_llm_moderation_v2.R

suppressMessages({
  library(dplyr)
  library(brms)
  library(bayestestR)
})

out_dir <- "../../output/5_inference_final"
model_dir <- file.path(out_dir, "models_bayesian")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)
topic_path <- "../../input/topics/topic_labels.csv"
cov_dir <- "../../input/covariates"
MECHS <- strsplit(Sys.getenv("MECHS", "ER,IP,EX"), ",")[[1]]

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
GROUP_LEVELS <- c("real", as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_")))
RESIDUAL_TOPICS <- c("Nonmedical", "Medical; other")   # original labels
EXPLORATORY_BELOW <- 0.05                              # share of all threads

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
LLMS <- c(gemini = "gem", openai = "opn", grok = "grk")

CHAINS <- 4
THREADS_PER_CHAIN <- 4
ITER <- 2000
WARMUP <- 1000
SEED <- 20260101
ADAPT_DELTA <- 0.95
ITER_OVERRIDES <- c(v2_ER_D2_cancer = 4000, v2_ER_F_severity = 4000, v2_IP_C2_topic_gap = 4000, v2_IP_D2_cancer = 4000,
                      v2_EX_C2_topic_gap = 4000, v2_EX_D2_cancer = 4000, v2_EX_F_severity = 4000)   # e.g. c(v2_IP_E_topic_by_llm = 4000); brm's file cache ignores iter changes, so move the old .rds aside first
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
    gem = as.integer(source == "synthetic" & model %in% "gemini"),
    opn = as.integer(source == "synthetic" & model %in% "openai"),
    grk = as.integer(source == "synthetic" & model %in% "grok"),
    log_length = log(pmax(lengths(strsplit(as.character(response_post), "\\s+")), 1)),
    ER_bin = as.integer(ER_label >= 1),
    IP_bin = as.integer(IP_label >= 1),
    EX_bin = as.integer(EX_label >= 1)
  )

# response weights (as 3c/3h): computed on all threads before any filtering
real_count <- results %>% filter(source == "real") %>% count(post_id, name = "real_count")
results <- results %>%
  left_join(real_count, by = "post_id") %>%
  mutate(real_count = coalesce(real_count, 0L)) %>%
  group_by(post_id, model, method) %>%
  mutate(cell_n = n()) %>%
  ungroup() %>%
  mutate(w = ifelse(source == "real", 1, pmin(1, real_count / cell_n))) %>%
  filter(w > 0)

# topic labels + covariates, one row per thread
topics <- read.csv(topic_path, stringsAsFactors = FALSE) %>%
  mutate(cancer_relevant = case_when(tolower(trimws(cancer_relevant)) == "true" ~ 1L,
                                     tolower(trimws(cancer_relevant)) == "false" ~ 0L,
                                     TRUE ~ NA_integer_)) %>%
  select(submission_id, discipline, cancer_relevant)
demo <- read.csv(file.path(cov_dir, "poster_demographics.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, subject, age_years, sex)
sev <- read.csv(file.path(cov_dir, "severity.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, severity_score)
roles <- read.csv(file.path(cov_dir, "first_responder_roles.csv"), stringsAsFactors = FALSE) %>%
  select(submission_id, specialty_kind, specialty_discipline)
stopifnot(!anyDuplicated(topics$submission_id), !anyDuplicated(demo$submission_id),
          !anyDuplicated(sev$submission_id), !anyDuplicated(roles$submission_id),
          sum(topics$cancer_relevant, na.rm = TRUE) == 518, sum(is.na(topics$cancer_relevant)) == 2)

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
  left_join(threads %>% select(submission_id, discipline, topic_label, cancer_relevant, match3,
                               subject, sex3, age10, age_na, sev_c, sev_na),
            by = "submission_id") %>%
  mutate(real_match = as.integer(source == "real" & match3 == "match"),
         real_mismatch = as.integer(source == "real" & match3 == "mismatch"),
         topic = topic_code(topic_label))
stopifnot(!anyNA(results$discipline),
          !anyNA(results[, c("subject", "sex3", "age10", "age_na", "sev_c", "sev_na", "real_match", "real_mismatch")]))

n_threads_all <- n_distinct(results$submission_id)
topic_lookup <- results %>%
  distinct(submission_id, topic, topic_label, discipline) %>%
  filter(!discipline %in% RESIDUAL_TOPICS) %>%
  count(topic, topic_label, name = "n_threads") %>%
  mutate(share = n_threads / n_threads_all, exploratory = share < EXPLORATORY_BELOW)

dat_topic <- results %>% filter(!discipline %in% RESIDUAL_TOPICS) %>% mutate(topic = factor(topic))
dat_cancer <- results %>% filter(!is.na(cancer_relevant))
stopifnot(n_distinct(dat_topic$topic) == 11)

cat(sprintf("all: %d rows / %d threads | topic models: %d rows / %d threads / %d topics | cancer model: %d rows / %d threads\n",
            nrow(results), n_threads_all, nrow(dat_topic), n_distinct(dat_topic$post_id), n_distinct(dat_topic$topic),
            nrow(dat_cancer), n_distinct(dat_cancer$post_id)))
cat("first-responder match among threads:", paste(names(table(threads$match3)), table(threads$match3), collapse = ", "), "\n")

md5 <- function(f) unname(tools::md5sum(f))
writeLines(c(
  sprintf("date: %s", Sys.time()), sprintf("seed: %d", SEED),
  sprintf("chains: %d, iter: %d, warmup: %d, threads_per_chain: %d, adapt_delta: %s", CHAINS, ITER, WARMUP, THREADS_PER_CHAIN, ADAPT_DELTA),
  sprintf("iteration overrides: %s", if (length(ITER_OVERRIDES)) paste(names(ITER_OVERRIDES), ITER_OVERRIDES, sep = " = ", collapse = ", ") else "none"),
  sprintf("topic file md5: %s", md5(topic_path)),
  sprintf("covariate md5: demographics %s, severity %s, roles %s",
          md5(file.path(cov_dir, "poster_demographics.csv")), md5(file.path(cov_dir, "severity.csv")),
          md5(file.path(cov_dir, "first_responder_roles.csv"))),
  sprintf("covariate terms: %s", COV),
  "weights: synthetic w = min(1, real_count / cell_n), real w = 1",
  sprintf("cmdstan_version: %s", as.character(cmdstanr::cmdstan_version())),
  "", "sessionInfo():", capture.output(sessionInfo())
), file.path(model_dir, "run_metadata_v2.txt"))

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
priors_common <- c(
  prior_string("normal(0, 2.5)", class = "b"),
  prior_string("student_t(3, 0, 2.5)", class = "Intercept")
)
priors_topic <- c(priors_common,
  prior_string("student_t(3, 0, 2.5)", class = "sd", group = "post_id"),
  prior_string("normal(0, 1)", class = "sd", group = "topic"))
priors_llm <- c(priors_topic, prior_string("lkj(2)", class = "cor", group = "topic"))
priors_cancer <- c(priors_common, prior_string("student_t(3, 0, 2.5)", class = "sd"))

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
report_conv <- function(label, cv) {
  cat(sprintf("  %s converged: %s (max Rhat %.4f, min ESS ratio %.3f, divergences %s)\n",
              label, cv$converged, cv$max_rhat, cv$min_neff_ratio, cv$divergent_transitions))
}

summ <- function(x) {
  h <- bayestestR::hdi(x, ci = 0.95)
  c(median = median(x), lcl = h$CI_low, ucl = h$CI_high, pd = max(mean(x > 0), mean(x < 0)))
}
summ_or <- function(x, prefix) {
  s <- summ(x)
  setNames(c(exp(s[["median"]]), exp(s[["lcl"]]), exp(s[["ucl"]]), s[["pd"]]),
           paste0(prefix, c("_or", "_lcl", "_ucl", "_pd")))
}
sd_row <- function(x, mech, parameter, model) {
  h <- bayestestR::hdi(x, ci = 0.95)
  data.frame(mechanism = mech, model = model, parameter = parameter, median = median(x), lcl = h$CI_low, ucl = h$CI_high)
}

# per-topic gap and shift for one random-slope column; base_gap = the average
# (over groups) synthetic-vs-real log-odds gap that the slope deviates from
topic_effects <- function(dr, slope, base_gap, mech, who) {
  pat <- sprintf("^r_topic\\[(.*),%s\\]$", slope)
  cols <- grep(pat, names(dr), value = TRUE)
  stopifnot(length(cols) == n_distinct(dat_topic$topic))
  bind_rows(lapply(cols, function(cn) {
    dev <- dr[[cn]]
    c(topic = sub(pat, "\\1", cn), as.list(summ_or(base_gap + dev, "gap")), as.list(summ_or(dev, "shift")))
  })) %>%
    mutate(across(-topic, as.numeric), mechanism = mech, who = who) %>%
    left_join(topic_lookup, by = "topic") %>%
    mutate(shift_credible = shift_pd > CREDIBLE_PD & (shift_lcl > 1 | shift_ucl < 1))
}

run_mechanism <- function(mech) {
  resp <- paste0(mech, "_bin")
  cat(sprintf("\n=== %s (response %s) ===\n", mech, resp))

  # ---- C2: pooled topic moderation + covariates ----
  cat(sprintf("--- %s: Model C2 (topic, pooled AI, n=%d) ---\n", mech, nrow(dat_topic)))
  fC <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + %s + log_length + (1 + synthetic | topic) + (1 | post_id)", resp, COV)),
                 dat_topic, sprintf("v2_%s_C2_topic_gap", mech), priors_topic)
  convC <- check_convergence_bayes(fC); report_conv("C2", convC)
  dr <- as.data.frame(brms::as_draws_df(fC))
  b_syn <- dr[, grep("^b_group", names(dr)), drop = FALSE]; stopifnot(ncol(b_syn) == 9)
  avg_gap <- rowMeans(b_syn)
  topic_tbl <- topic_effects(dr, "synthetic", avg_gap, mech, "all AI (pooled)")
  overall <- data.frame(who = "all AI (pooled)", t(summ_or(avg_gap, "gap")), mechanism = mech)
  cov_cols <- grep("^b_(subject|age10|age_na|sex3|sev_c|sev_na|real_match|real_mismatch)", names(dr), value = TRUE)
  cov_tbl <- bind_rows(lapply(cov_cols, function(cn) {
    c(list(term = sub("^b_", "", cn)), as.list(summ_or(dr[[cn]], "effect")))
  })) %>% mutate(across(-term, as.numeric), mechanism = mech)
  sd_tbl <- bind_rows(sd_row(dr$sd_topic__Intercept, mech, "topic_intercept_sd", "C2"),
                      sd_row(dr$sd_topic__synthetic, mech, "topic_slope_sd", "C2"),
                      sd_row(dr$sd_post_id__Intercept, mech, "post_id_sd", "C2"))

  # ---- D2: cancer relevance + covariates ----
  cat(sprintf("--- %s: Model D2 (cancer, n=%d) ---\n", mech, nrow(dat_cancer)))
  fD <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + cancer_relevant + synthetic:cancer_relevant + %s + log_length + (1 | post_id)", resp, COV)),
                 dat_cancer, sprintf("v2_%s_D2_cancer", mech), priors_cancer)
  convD <- check_convergence_bayes(fD); report_conv("D2", convD)
  dd <- as.data.frame(brms::as_draws_df(fD))
  gap0 <- rowMeans(dd[, grep("^b_group", names(dd)), drop = FALSE])
  int_col <- grep("^b_synthetic:cancer_relevant$|^b_cancer_relevant:synthetic$", names(dd), value = TRUE)
  stopifnot(length(int_col) == 1)
  cancer_tbl <- data.frame(mechanism = mech, t(c(
    summ_or(gap0, "gap_not_cancer"), summ_or(gap0 + dd[[int_col]], "gap_cancer"),
    summ_or(dd[[int_col]], "shift"), summ_or(dd$b_cancer_relevant, "cancer_main"))))

  # ---- E: topic moderation separately for each LLM ----
  cat(sprintf("--- %s: Model E (topic x LLM, n=%d) ---\n", mech, nrow(dat_topic)))
  fE <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + %s + log_length + (1 + gem + opn + grk | topic) + (1 | post_id)", resp, COV)),
                 dat_topic, sprintf("v2_%s_E_topic_by_llm", mech), priors_llm)
  convE <- check_convergence_bayes(fE); report_conv("E", convE)
  de <- as.data.frame(brms::as_draws_df(fE))
  llm_gap <- lapply(names(LLMS), function(L) {
    cols <- grep(paste0("^b_group", L, "_"), names(de), value = TRUE); stopifnot(length(cols) == 3)
    rowMeans(de[, cols, drop = FALSE])
  }); names(llm_gap) <- names(LLMS)
  by_llm <- bind_rows(lapply(names(LLMS), function(L) topic_effects(de, LLMS[[L]], llm_gap[[L]], mech, L)))
  llm_overall <- bind_rows(lapply(names(LLMS), function(L) data.frame(who = L, t(summ_or(llm_gap[[L]], "gap")), mechanism = mech)))

  # topic-specific differences between LLMs: ratio of ORs (gap) and of shifts
  pairs <- list(c("gemini", "openai"), c("gemini", "grok"), c("openai", "grok"))
  shift_draws <- function(L) { cols <- grep(sprintf("^r_topic\\[(.*),%s\\]$", LLMS[[L]]), names(de), value = TRUE)
                               setNames(lapply(cols, function(cn) de[[cn]]), sub(sprintf("^r_topic\\[(.*),%s\\]$", LLMS[[L]]), "\\1", cols)) }
  sh <- lapply(names(LLMS), shift_draws); names(sh) <- names(LLMS)
  diff_tbl <- bind_rows(lapply(pairs, function(p) {
    bind_rows(lapply(names(sh[[p[1]]]), function(tc) {
      d_shift <- sh[[p[1]]][[tc]] - sh[[p[2]]][[tc]]
      d_gap <- (llm_gap[[p[1]]] + sh[[p[1]]][[tc]]) - (llm_gap[[p[2]]] + sh[[p[2]]][[tc]])
      c(topic = tc, pair = paste(p, collapse = " vs "), as.list(summ_or(d_gap, "gap_ratio")), as.list(summ_or(d_shift, "shift_ratio")))
    }))
  })) %>% mutate(across(-c(topic, pair), as.numeric), mechanism = mech) %>%
    left_join(topic_lookup, by = "topic") %>%
    mutate(shift_ratio_credible = shift_ratio_pd > CREDIBLE_PD & (shift_ratio_lcl > 1 | shift_ratio_ucl < 1))

  cor_cols <- grep("^cor_topic__(gem|opn|grk)__(gem|opn|grk)$", names(de), value = TRUE)
  sd_llm <- bind_rows(
    sd_row(de$sd_topic__Intercept, mech, "topic_intercept_sd", "E"),
    # wrapped in their own bind_rows: a NAMED list of data frames passed alongside other
    # arguments is read as one wide table (this produced all-NA rows in the first test run)
    bind_rows(lapply(unname(LLMS), function(s) sd_row(de[[paste0("sd_topic__", s)]], mech, paste0("topic_slope_sd_", s), "E"))),
    sd_row(de$sd_post_id__Intercept, mech, "post_id_sd", "E"),
    bind_rows(lapply(cor_cols, function(cn) sd_row(de[[cn]], mech, sub("^cor_topic__", "topic_slope_cor_", cn), "E"))))

  # ---- F: severity as a moderator of each LLM's gap, net of topic ----
  cat(sprintf("--- %s: Model F (severity moderation, n=%d) ---\n", mech, nrow(dat_topic)))
  fF <- fit_brms(as.formula(sprintf("%s | weights(w) ~ group + %s + gem:sev_c + opn:sev_c + grk:sev_c + synthetic:sev_na + log_length + (1 + synthetic | topic) + (1 | post_id)", resp, COV)),
                 dat_topic, sprintf("v2_%s_F_severity", mech), priors_topic)
  convF <- check_convergence_bayes(fF); report_conv("F", convF)
  df_ <- as.data.frame(brms::as_draws_df(fF))
  slope_col <- function(a, b) {
    cols <- grep(sprintf("^b_(%s:%s|%s:%s)$", a, b, b, a), names(df_), value = TRUE)
    stopifnot(length(cols) == 1); cols
  }
  sev_slopes <- lapply(LLMS, function(s) df_[[slope_col(s, "sev_c")]])   # keyed gemini/openai/grok
  sev_tbl <- bind_rows(
    lapply(names(LLMS), function(L) data.frame(who = L, term = "extra change in gap per severity level",
                                               t(summ_or(sev_slopes[[L]], "sev")), mechanism = mech)),
    data.frame(who = "all AI (mean of the three)", term = "extra change in gap per severity level",
               t(summ_or(Reduce(`+`, sev_slopes) / 3, "sev")), mechanism = mech),
    data.frame(who = "all AI", term = "severity not rated: shift in gap",
               t(summ_or(df_[[slope_col("synthetic", "sev_na")]], "sev")), mechanism = mech))

  list(
    convergence = bind_rows(cbind(mechanism = mech, model = "C2_topic_gap", convC),
                            cbind(mechanism = mech, model = "D2_cancer", convD),
                            cbind(mechanism = mech, model = "E_topic_by_llm", convE),
                            cbind(mechanism = mech, model = "F_severity", convF)),
    by_topic = bind_rows(topic_tbl, overall), sd = sd_tbl, cancer = cancer_tbl, covariates = cov_tbl,
    by_llm = bind_rows(by_llm, llm_overall), diffs = diff_tbl, sd_llm = sd_llm, severity = sev_tbl
  )
}

res <- setNames(lapply(MECHS, run_mechanism), MECHS)
pick <- function(nm) bind_rows(lapply(res, `[[`, nm))

write.csv(pick("convergence"), file.path(out_dir, "v2_topic_convergence.csv"), row.names = FALSE)
write.csv(pick("by_topic"), file.path(out_dir, "v2_topic_gap_by_topic.csv"), row.names = FALSE)
write.csv(pick("sd"), file.path(out_dir, "v2_topic_sd_summary.csv"), row.names = FALSE)
write.csv(pick("cancer"), file.path(out_dir, "v2_topic_cancer_moderation.csv"), row.names = FALSE)
write.csv(pick("covariates"), file.path(out_dir, "v2_covariate_effects.csv"), row.names = FALSE)
write.csv(pick("by_llm"), file.path(out_dir, "v2_topic_by_llm.csv"), row.names = FALSE)
write.csv(pick("diffs"), file.path(out_dir, "v2_llm_differences_by_topic.csv"), row.names = FALSE)
write.csv(pick("sd_llm"), file.path(out_dir, "v2_llm_sd_summary.csv"), row.names = FALSE)
write.csv(pick("severity"), file.path(out_dir, "v2_severity_moderation.csv"), row.names = FALSE)
write.csv(topic_lookup, file.path(out_dir, "v2_topic_lookup.csv"), row.names = FALSE)

cat("\n=== Convergence ===\n"); print(pick("convergence"))
cat("\n=== SD of topic effects (pooled) ===\n"); print(pick("sd"))
cat("\n=== SD / correlations of topic effects (by LLM) ===\n"); print(pick("sd_llm"))
cat("\n=== Cancer relevance ===\n"); print(pick("cancer"))
cat("\n=== Covariate effects ===\n"); print(pick("covariates"))
cat("\n=== Severity moderation of the gap ===\n"); print(pick("severity"))
cat("\nWrote v2_*.csv to", out_dir, "\n")
