#!/usr/bin/env Rscript
#
# Joint tests for the hypotheses on AI model, prompting strategy and their interaction
# (paper_empirical_section.Rmd). Reads the saved posterior fits, writes small CSVs.
#
# For a vector d of k contrasts with posterior draws, the *joint probability of direction* is
# the posterior probability that a draw lies closer to the posterior mean (in Mahalanobis distance)
# than the point "no difference" (d = 0) does:  jpd = pchisq(W, k),  W = mean' Cov^-1 mean.
# jpd > 0.975 means that 0 lies outside the 95% joint credible region; for k = 1 it reduces to the
# probability of direction used elsewhere (two-sided: jpd > 0.975 <=> pd > 0.975 with the usual caveat).
#
#   model        H2: the gap differs between AI models      (2 contrasts of model means over strategies)
#   strategy     H3: the gap differs between strategies     (2 contrasts of strategy means over models)
#   interaction  H4: strategy effects differ between models (4 contrasts-of-contrasts)
#   severity     H7: the gap changes with severity          (3 model-specific severity slopes = 0; v2 model F)
#
# Because the gap of every AI group is measured against the same real replies within the same threads,
# the real replies cancel out of any contrast between AI groups, so H2 to H4 are comparisons between AI groups (RQ2).
# H5 (RQ3: AI versus real in each of the nine groups) uses the group gaps themselves (v3_group_effects.csv).
#
# Run from src/5_inference_final:  Rscript paper_joint_tests.R
# Experts-only sensitivity run (see paper_config.R):  EXCLUDE_LAYPEOPLE=1 Rscript paper_joint_tests.R

suppressMessages({ library(brms); library(bayestestR) })
source("paper_config.R")
out_dir <- OUT_DIR
mod_dir <- file.path(out_dir, "models_bayesian")
MODELS <- c("gemini", "openai", "grok"); STRATS <- c("dg", "sag", "sagii")
mechs <- c("ER", "IP", "EX")

jt <- function(D) {
  D <- as.matrix(D); k <- ncol(D); mu <- colMeans(D); S <- cov(D)
  W <- drop(t(mu) %*% solve(S) %*% mu)
  c(df = k, W = W, jpd = pchisq(W, k))
}
cri <- function(x) { h <- hdi(x, ci = 0.95); c(or = exp(median(x)), lcl = exp(h$CI_low), ucl = exp(h$CI_high), pd = max(mean(x > 0), mean(x < 0))) }

tests <- list(); pairs <- list()
for (version in c("weighted", "thinned")) {
  prefix <- if (version == "weighted") "v3_" else "sub1_v3_"
  for (m in mechs) {
    d <- as.data.frame(as_draws_df(readRDS(file.path(mod_dir, sprintf("%s%s_L1_length.rds", prefix, m)))))
    beta <- function(mo, st) d[[paste0("b_group", mo, "_", st)]]
    mm <- function(mo) rowMeans(sapply(STRATS, function(s) beta(mo, s)))
    ss <- function(st) rowMeans(sapply(MODELS, function(mo) beta(mo, st)))
    inter <- function(m1, m2, s1, s2) (beta(m1, s1) - beta(m1, s2)) - (beta(m2, s1) - beta(m2, s2))
    tests[[length(tests) + 1]] <- data.frame(version, mechanism = m, hypothesis = "model",
      t(jt(cbind(mm("gemini") - mm("openai"), mm("openai") - mm("grok")))))
    tests[[length(tests) + 1]] <- data.frame(version, mechanism = m, hypothesis = "strategy",
      t(jt(cbind(ss("dg") - ss("sag"), ss("sag") - ss("sagii")))))
    tests[[length(tests) + 1]] <- data.frame(version, mechanism = m, hypothesis = "interaction",
      t(jt(cbind(inter("gemini", "openai", "dg", "sag"), inter("gemini", "openai", "sag", "sagii"),
                 inter("openai", "grok", "dg", "sag"), inter("openai", "grok", "sag", "sagii")))))
    for (mp in list(c("gemini", "openai"), c("gemini", "grok"), c("openai", "grok")))
      for (sp in list(c("dg", "sag"), c("dg", "sagii"), c("sag", "sagii")))
        pairs[[length(pairs) + 1]] <- data.frame(version, mechanism = m, model_pair = paste(mp, collapse = " vs "),
          strategy_pair = paste(sp, collapse = " vs "), t(cri(inter(mp[1], mp[2], sp[1], sp[2]))))
    rm(d)
  }
}
# severity (weighted fits of model F only)
for (m in mechs) {
  d <- as.data.frame(as_draws_df(readRDS(file.path(mod_dir, sprintf("v2_%s_F_severity.rds", m)))))
  tests[[length(tests) + 1]] <- data.frame(version = "weighted", mechanism = m, hypothesis = "severity",
    t(jt(d[, c("b_sev_c:gem", "b_sev_c:opn", "b_sev_c:grk")])))
}
tests <- do.call(rbind, tests); pairs <- do.call(rbind, pairs)
write.csv(tests, file.path(out_dir, "paper_joint_tests.csv"), row.names = FALSE)
write.csv(pairs, file.path(out_dir, "paper_interaction_pairs.csv"), row.names = FALSE)
print(tests, digits = 4); cat("\nCredible pairwise interaction contrasts:\n")
print(aggregate(cbind(credible = pd > 0.975 & (lcl > 1 | ucl < 1)) ~ version + mechanism, data = pairs, FUN = sum))
