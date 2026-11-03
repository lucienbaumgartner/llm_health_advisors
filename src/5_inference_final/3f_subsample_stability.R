#!/usr/bin/env Rscript
#
# Stability check for the per-thread balancing step (no outlier exclusion):
# how much do the model estimates change depending on WHICH random subsample
# of synthetic comments is drawn? Refits 3b's Interpretations and Explorations
# glmer models (fast; ER's clmm is far too slow to repeat) on many independent
# balanced subsamples and compares
#   - between-subsample SD of each contrast estimate (log-odds-ratio scale)
#   - to the within-fit standard error of that contrast.
# If SD << SE, subsampling noise is negligible next to ordinary sampling
# uncertainty and one subsample is a defensible choice: the total variance is
# approximately SE^2 + SD^2, so a ratio SD/SE of 0.3 inflates variance by only
# ~9%. Contrasts are on the link (log-odds) scale.
#
# Balancing here: for each (post_id, model, method) cell, keep a random
# min(cell size, that thread's real comment count) synthetic comments; real
# comments are never touched. Subsampling uses an explicit per-replicate seed
# (BASE_SEED + i) so results don't depend on how work is scheduled across cores.
#
# Usage: N_SEEDS=50 N_CORES=12 Rscript 3f_subsample_stability.R

suppressMessages({
  library(dplyr)
  library(lme4)
  library(emmeans)
  library(parallel)
})

out_dir <- "../../output/5_inference_final"
N_SEEDS <- as.integer(Sys.getenv("N_SEEDS", "50"))
N_CORES <- as.integer(Sys.getenv("N_CORES", "12"))
BASE_SEED <- 20260101

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
GROUP_LEVELS <- c("real", as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_")))

cat("Loading data...\n")
results <- read.csv(file.path(out_dir, "inference_results.csv"))
results <- results %>%
  filter(!is.na(response_post) & response_post != "") %>%
  mutate(
    group = factor(ifelse(source == "real", "real", paste(model, method, sep = "_")), levels = GROUP_LEVELS),
    post_id = factor(post_id),
    log_length = log(pmax(lengths(strsplit(as.character(response_post), "\\s+")), 1)),
    IP_bin = as.integer(IP_label == 2),
    EX_bin = as.integer(EX_label == 2)
  ) %>%
  select(post_id, source, model, method, group, log_length, IP_bin, EX_bin)

real <- results %>% filter(source == "real")
real_count <- real %>% count(post_id, name = "real_count")
synth <- results %>%
  filter(source == "synthetic") %>%
  mutate(model = factor(model, levels = MODEL_LEVELS), method = factor(method, levels = METHOD_LEVELS)) %>%
  left_join(real_count, by = "post_id") %>%
  mutate(real_count = ifelse(is.na(real_count), 0L, real_count))

cat(sprintf("real = %d, synthetic (pre-balancing) = %d, posts = %d; replicates = %d on %d cores\n",
            nrow(real), nrow(synth), n_distinct(results$post_id), N_SEEDS, N_CORES))

tidy_ct <- function(ct, stage, mech, seed) {
  s <- as.data.frame(summary(ct))
  s$cond <- if ("method" %in% names(s)) as.character(s$method) else if ("model" %in% names(s)) as.character(s$model) else NA_character_
  s$contrast <- as.character(s$contrast)
  data.frame(seed = seed, mechanism = mech, stage = stage, cond = s$cond, contrast = s$contrast,
             estimate = s$estimate, SE = s$SE, stringsAsFactors = FALSE)
}

fit_glmer <- function(response, rhs, data) {
  f <- as.formula(sprintf("%s ~ %s + (1|post_id)", response, rhs))
  warn <- character()
  fit <- withCallingHandlers(
    glmer(f, data = data, family = binomial, control = glmerControl(optimizer = "bobyqa")),
    warning = function(w) { warn <<- c(warn, conditionMessage(w)); invokeRestart("muffleWarning") }
  )
  msgs <- fit@optinfo$conv$lme4$messages
  list(fit = fit, flagged = length(warn) > 0 || !is.null(msgs))
}

run_seed <- function(i) {
  t0 <- Sys.time()
  set.seed(BASE_SEED + i)
  synth_bal <- synth %>%
    group_by(post_id, model, method) %>%
    mutate(rk = rank(runif(n()), ties.method = "first")) %>%
    ungroup() %>%
    filter(rk <= real_count)
  dat_all <- bind_rows(real %>% select(post_id, group, log_length, IP_bin, EX_bin),
                       synth_bal %>% select(post_id, group, log_length, IP_bin, EX_bin))
  out <- list(); flags <- list()
  for (mech in c("IP", "EX")) {
    resp <- paste0(mech, "_bin")

    a <- fit_glmer(resp, "group + log_length", dat_all)
    ctA <- contrast(emmeans(a$fit, ~group), "trt.vs.ctrl", adjust = "none")
    out[[length(out) + 1]] <- tidy_ct(ctA, "A_vs_real", mech, i)

    b <- fit_glmer(resp, "model * method + log_length", synth_bal)
    ctB1 <- contrast(emmeans(b$fit, ~ model | method), "pairwise", adjust = "none")
    ctB2 <- contrast(emmeans(b$fit, ~ method | model), "pairwise", adjust = "none")
    out[[length(out) + 1]] <- tidy_ct(ctB1, "B_between_model", mech, i)
    out[[length(out) + 1]] <- tidy_ct(ctB2, "B_within_model", mech, i)

    flags[[length(flags) + 1]] <- data.frame(seed = i, mechanism = mech, flagged_A = a$flagged, flagged_B = b$flagged)
  }
  cat(sprintf("replicate %d done (%.0fs, n_all = %d, n_synth = %d)\n", i,
              as.numeric(Sys.time() - t0, units = "secs"), nrow(dat_all), nrow(synth_bal)))
  list(est = bind_rows(out), flags = bind_rows(flags))
}

res <- mclapply(seq_len(N_SEEDS), function(i) tryCatch(run_seed(i), error = function(e) {
  cat(sprintf("replicate %d FAILED: %s\n", i, conditionMessage(e))); NULL
}), mc.cores = N_CORES, mc.preschedule = FALSE)

ok <- !vapply(res, is.null, logical(1))
cat(sprintf("\n%d / %d replicates succeeded\n", sum(ok), N_SEEDS))
est <- bind_rows(lapply(res[ok], `[[`, "est"))
flags <- bind_rows(lapply(res[ok], `[[`, "flags"))
write.csv(est, file.path(out_dir, "stability_subsample_estimates.csv"), row.names = FALSE)

summary_tbl <- est %>%
  mutate(z = estimate / SE) %>%
  group_by(mechanism, stage, cond, contrast) %>%
  summarise(n_reps = n(), mean_estimate = mean(estimate), sd_estimate = sd(estimate), mean_SE = mean(SE),
            sd_over_se = sd_estimate / mean_SE, variance_inflation = 1 + sd_over_se^2,
            prop_sig_05 = mean(abs(z) > 1.96),
            prop_same_sign = mean(sign(estimate) == sign(mean_estimate)), .groups = "drop")
write.csv(summary_tbl, file.path(out_dir, "stability_subsample_summary.csv"), row.names = FALSE)

cat("\n=== SD across subsamples / mean within-fit SE, by mechanism and stage ===\n")
print(summary_tbl %>% group_by(mechanism, stage) %>%
        summarise(n_contrasts = n(), median_ratio = median(sd_over_se), max_ratio = max(sd_over_se),
                  max_variance_inflation = max(variance_inflation), .groups = "drop") %>% as.data.frame(), digits = 3)
cat("\n=== Contrasts whose nominal significance (|z| > 1.96) is not the same in every replicate ===\n")
print(summary_tbl %>% filter(prop_sig_05 > 0, prop_sig_05 < 1) %>%
        select(mechanism, stage, cond, contrast, mean_estimate, mean_SE, sd_estimate, prop_sig_05) %>% as.data.frame(), digits = 3)
cat("\n=== Fits flagged with a lme4 warning/convergence message (count of mechanism x replicate) ===\n")
print(flags %>% group_by(mechanism) %>% summarise(n = n(), flagged_A = sum(flagged_A), flagged_B = sum(flagged_B)) %>% as.data.frame())
cat("\nWrote stability_subsample_estimates.csv and stability_subsample_summary.csv to", out_dir, "\n")
