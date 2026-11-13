#!/usr/bin/env Rscript
#
# Weighted full-data alternative to per-thread subsampling. Instead of randomly
# keeping min(n, real_count) synthetic comments per (post_id, model, method)
# cell, keep ALL synthetic comments and give each weight min(1, real_count / n)
# (real comments weight 1). The expected subsampled log-likelihood over the
# random draw equals this weighted log-likelihood, so one weighted fit targets
# the average over subsamples without depending on any single draw.
#
# Fits the same Interpretations / Explorations glmer models as 3b/3f and
# compares them with the 50-subsample results from 3f_subsample_stability.R
# (stability_subsample_summary.csv): are the weighted estimates close to the
# subsample average, and how do the standard errors compare? Contrasts are on
# the link (log-odds) scale. Note glmer treats weights as precision weights,
# so the weighted SEs are naive (no sandwich correction) - compared, not
# assumed valid.

suppressMessages({
  library(dplyr)
  library(lme4)
  library(emmeans)
  library(parallel)
})

out_dir <- "../../output/5_inference_final"
MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
GROUP_LEVELS <- c("real", as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_")))

results <- read.csv(file.path(out_dir, "inference_results.csv")) %>%
  filter(!is.na(response_post) & response_post != "") %>%
  mutate(
    group = factor(ifelse(source == "real", "real", paste(model, method, sep = "_")), levels = GROUP_LEVELS),
    post_id = factor(post_id),
    log_length = log(pmax(lengths(strsplit(as.character(response_post), "\\s+")), 1)),
    IP_bin = as.integer(IP_label == 2),
    EX_bin = as.integer(EX_label == 2)
  ) %>%
  select(post_id, source, model, method, group, log_length, IP_bin, EX_bin)

real_count <- results %>% filter(source == "real") %>% count(post_id, name = "real_count")
dat <- results %>%
  left_join(real_count, by = "post_id") %>%
  mutate(real_count = ifelse(is.na(real_count), 0L, real_count)) %>%
  group_by(post_id, model, method) %>%
  mutate(cell_n = n()) %>%
  ungroup() %>%
  mutate(w = ifelse(source == "real", 1, pmin(1, real_count / cell_n))) %>%
  filter(w > 0)

synth <- dat %>% filter(source == "synthetic") %>%
  mutate(model = factor(model, levels = MODEL_LEVELS), method = factor(method, levels = METHOD_LEVELS))

cat(sprintf("rows: %d (all), %d (synthetic); sum of synthetic weights = %.0f (subsampled n = 77603)\n",
            nrow(dat), nrow(synth), sum(synth$w)))

tidy_ct <- function(ct, stage, mech) {
  s <- as.data.frame(summary(ct))
  cond <- if ("method" %in% names(s)) as.character(s$method) else if ("model" %in% names(s)) as.character(s$model) else NA_character_
  data.frame(mechanism = mech, stage = stage, cond = cond, contrast = as.character(s$contrast),
             weighted_estimate = s$estimate, weighted_SE = s$SE, stringsAsFactors = FALSE)
}

fit_w <- function(resp, rhs, data) {
  glmer(as.formula(sprintf("%s ~ %s + (1|post_id)", resp, rhs)), data = data, weights = w,
        family = binomial, control = glmerControl(optimizer = "bobyqa"))
}

run_mech <- function(mech) {
  resp <- paste0(mech, "_bin")
  t0 <- Sys.time()
  a <- fit_w(resp, "group + log_length", dat)
  b <- fit_w(resp, "model * method + log_length", synth)
  cat(sprintf("%s fitted (%.0fs)\n", mech, as.numeric(Sys.time() - t0, units = "secs")))
  bind_rows(
    tidy_ct(contrast(emmeans(a, ~group), "trt.vs.ctrl", adjust = "none"), "A_vs_real", mech),
    tidy_ct(contrast(emmeans(b, ~ model | method), "pairwise", adjust = "none"), "B_between_model", mech),
    tidy_ct(contrast(emmeans(b, ~ method | model), "pairwise", adjust = "none"), "B_within_model", mech)
  )
}

weighted <- bind_rows(mclapply(c("IP", "EX"), run_mech, mc.cores = 2))

sub <- read.csv(file.path(out_dir, "stability_subsample_summary.csv")) %>%
  select(mechanism, stage, cond, contrast, subsample_mean_estimate = mean_estimate,
         subsample_sd = sd_estimate, subsample_mean_SE = mean_SE, prop_sig_subsamples = prop_sig_05)
cmp <- weighted %>% left_join(sub, by = c("mechanism", "stage", "cond", "contrast")) %>%
  mutate(weighted_z = weighted_estimate / weighted_SE,
         diff_in_subsample_sd = (weighted_estimate - subsample_mean_estimate) / subsample_sd,
         se_ratio_weighted_over_subsample = weighted_SE / subsample_mean_SE)
write.csv(cmp, file.path(out_dir, "stability_weighted_comparison.csv"), row.names = FALSE)

cat("\nn contrasts:", nrow(cmp), " unmatched:", sum(is.na(cmp$subsample_sd)), "\n")
print(cmp %>% group_by(mechanism, stage) %>%
        summarise(median_abs_diff_in_sd = median(abs(diff_in_subsample_sd)), max_abs_diff_in_sd = max(abs(diff_in_subsample_sd)),
                  median_se_ratio = median(se_ratio_weighted_over_subsample), .groups = "drop") %>% as.data.frame(), digits = 3)
cat("Wrote stability_weighted_comparison.csv\n")
