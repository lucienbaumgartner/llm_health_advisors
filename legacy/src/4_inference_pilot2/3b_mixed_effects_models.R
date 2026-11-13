#!/usr/bin/env Rscript
#
# Mixed-effects replacement for the Kruskal-Wallis omnibus test and the
# by-method/by-model pooled comparisons that used to live in
# 3_analyze_empathy_differences.py. Neither of those nonparametric tests
# accounted for comments clustering by post (multiple real/synthetic
# comments on the same thread aren't independent observations), and neither
# could control for comment length directly.
#
# Two models per mechanism, both with a post_id random intercept and
# log(length) as a covariate:
#
#   A. label ~ group + log_length + (1|post_id)              [all data]
#      group = real + 9 (model, method) combos, real as reference.
#      -> omnibus test for `group` (replacing Kruskal-Wallis), plus
#      model-based contrasts of each synthetic group vs. real (odds ratios),
#      as a model-adjusted complement to the existing Mann-Whitney U table
#      in 3_analyze_empathy_differences.py (not replaced).
#
#   B. label ~ model * method + log_length + (1|post_id)     [synthetic only]
#      "real" has no model/method to cross, so this is fit on synthetic
#      comments only. Gives model main effect, method main effect, their
#      interaction, and (since the interaction is significant for every
#      mechanism - pooled main-effect means would be misleading) two sets of
#      simple-effects contrasts: between models within each method, and
#      between methods within each model.
#
# Emotional Reactions (ER) uses all 3 empathy levels -> ordinal::clmm.
# Interpretations (IP) and Explorations (EX) never predict label 1 in this
# data (0/62161 occurrences for both, checked directly) - effectively binary
# outcomes, so they use lme4::glmer(family=binomial) on I(label==2) rather
# than a degenerate 3-level ordinal model.
#
# ER's model x method interaction has a real separation risk: grok_sagii has
# 0/6475 comments at label 2 (grok_dg: 1/6475, grok_sag: 5/6475). Convergence
# is checked and reported per fit rather than assumed clean.
#
# Effect tests use car::Anova(fit, type = 2) (Wald chi-square from a single
# fit) rather than refitting nested models for every term - clmm fits on the
# full ~62k-row / ~3000-post dataset are slow (tens of minutes), so a
# single-fit approach matters here. Falls back to a manual likelihood-ratio
# test (refit without the term) if car::Anova ever fails on a given fit.
#
# All contrasts are reported as odds ratios. For glmer this is emmeans'
# native type="response" transform. clmm has no single "response" scale for
# a 3-category cumulative link model, so emmeans just returns the raw
# latent-scale (log-odds) difference for it; exponentiating that by hand
# gives the same cumulative odds ratio a proportional-odds model is usually
# reported with - verified against a direct fit before writing this.
#
# Every fitted model is cached to output/4_inference_pilot2/models/*.rds -
# ER's clmm fits take ~15-35 minutes combined, so re-running this script
# after a change to the *output* side (a new contrast, a table tweak) reuses
# the cached fits instead of refitting. Delete the cache (or the specific
# .rds file) to force a refit after a change to the *model* formula/data.

suppressMessages({
  library(dplyr)
  library(ordinal)
  library(lme4)
  library(car)
  library(emmeans)
})

out_dir <- "../../output/4_inference_pilot2"
model_dir <- file.path(out_dir, "models")
dir.create(model_dir, showWarnings = FALSE, recursive = TRUE)

MODEL_LEVELS <- c("gemini", "openai", "grok")
METHOD_LEVELS <- c("dg", "sag", "sagii")
SYNTH_GROUPS <- as.vector(outer(MODEL_LEVELS, METHOD_LEVELS, paste, sep = "_"))
GROUP_LEVELS <- c("real", SYNTH_GROUPS)

cat("Loading data...\n")
results <- read.csv(file.path(out_dir, "inference_results.csv"))
results <- results %>% filter(!is.na(response_post) & response_post != "")

results <- results %>%
  mutate(
    group = factor(ifelse(source == "real", "real", paste(model, method, sep = "_")), levels = GROUP_LEVELS),
    post_id = factor(post_id),
    length = lengths(strsplit(as.character(response_post), "\\s+")),
    log_length = log(pmax(length, 1)),
    ER_f = factor(ER_label, levels = c(0, 1, 2), ordered = TRUE)
  )

synth <- results %>%
  filter(source == "synthetic") %>%
  mutate(model = factor(model, levels = MODEL_LEVELS),
         method = factor(method, levels = METHOD_LEVELS))

cat(sprintf("n = %d (all), %d (synthetic only), %d posts\n",
            nrow(results), nrow(synth), n_distinct(results$post_id)))

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

fit_or_load <- function(mechanism, stage, fit_fn) {
  path <- file.path(model_dir, sprintf("%s_%s.rds", mechanism, stage))
  if (file.exists(path)) {
    cat(sprintf("  loading cached fit: %s\n", path))
    return(readRDS(path))
  }
  t0 <- Sys.time()
  fit <- fit_fn()
  cat(sprintf("  fit time: %.1fs\n", as.numeric(Sys.time() - t0, units = "secs")))
  saveRDS(fit, path)
  fit
}

check_convergence <- function(fit) {
  if (inherits(fit, "clmm")) {
    max_grad <- suppressWarnings(as.numeric(fit$info$max.grad))
    ok <- is.finite(max_grad) && max_grad < 1e-3
    msg <- if (ok) NA_character_ else sprintf("max|grad| = %.3g (didn't fully settle)", max_grad)
    return(list(converged = ok, message = msg))
  }
  if (inherits(fit, "merMod")) {
    msgs <- fit@optinfo$conv$lme4$messages
    return(list(converged = is.null(msgs), message = if (is.null(msgs)) NA_character_ else paste(msgs, collapse = "; ")))
  }
  list(converged = NA, message = NA_character_)
}

# Wald chi-sq tests for every term via a single fit; falls back to a manual
# LRT (refit dropping the term) only if car::Anova errors out.
anova_table <- function(fit, mechanism, model_stage, formula_full, drop_fallback, data, fit_fn) {
  a <- tryCatch(car::Anova(fit, type = 2), error = function(e) e)
  if (!inherits(a, "error")) {
    return(data.frame(
      mechanism = mechanism, model_stage = model_stage, term = rownames(a),
      chisq = a[["Chisq"]], df = a[["Df"]], p_value = a[["Pr(>Chisq)"]],
      method = "wald_type2", row.names = NULL
    ))
  }
  cat(sprintf("  car::Anova failed (%s) - falling back to manual LRT for: %s\n",
              conditionMessage(a), drop_fallback))
  formula_reduced <- update(formula_full, paste(". ~ . -", drop_fallback))
  fit_reduced <- fit_fn(formula_reduced, data)
  lrt <- anova(fit_reduced, fit)
  lrt_row <- lrt[2, ]
  chisq <- if ("LR.stat" %in% names(lrt_row)) lrt_row[["LR.stat"]] else lrt_row[["Chisq"]]
  data.frame(
    mechanism = mechanism, model_stage = model_stage, term = drop_fallback,
    chisq = chisq, df = lrt_row[["Df"]] %||% lrt_row[["df"]], p_value = lrt_row[["Pr(>Chisq)"]],
    method = "lrt_fallback", row.names = NULL
  )
}

# Tidy an emmeans contrast object into a uniform data.frame: contrast (+ any
# conditioning column, e.g. "method" or "model" from a `~ X | Y` spec),
# odds_ratio, OR_LCL, OR_UCL, p_value. Odds ratios come free from emmeans for
# glmer (type="response"); for clmm we exponentiate the latent-scale
# estimate/CI ourselves (see module docstring).
tidy_contrast <- function(ct, use_binary) {
  if (use_binary) {
    df <- as.data.frame(summary(ct, infer = c(TRUE, TRUE), type = "response"))
    df$odds_ratio <- df$odds.ratio
    df$OR_LCL <- df$asymp.LCL
    df$OR_UCL <- df$asymp.UCL
  } else {
    df <- as.data.frame(summary(ct, infer = c(TRUE, TRUE)))
    df$odds_ratio <- exp(df$estimate)
    df$OR_LCL <- exp(df$asymp.LCL)
    df$OR_UCL <- exp(df$asymp.UCL)
    # clmm contrasts come back labeled "A - B" (a latent-scale difference) even
    # though we've exponentiated into a ratio - relabel to "A / B" so it
    # doesn't read as a subtraction, matching glmer's native ratio labels.
    df$contrast <- sub(" - ", " / ", df$contrast, fixed = TRUE)
  }
  df$p_value <- df$p.value
  keep_cols <- intersect(c("contrast", "method", "model", "odds_ratio", "OR_LCL", "OR_UCL", "p_value"), names(df))
  df[, keep_cols]
}

fit_mechanism <- function(mechanism, response_col, use_binary) {
  cat(sprintf("\n=== %s ===\n", mechanism))

  # --- Model A: group (real + 9 combos), all data ---
  cat(sprintf("--- %s: Model A (group, n=%d) ---\n", mechanism, nrow(results)))
  if (use_binary) {
    fA_formula <- as.formula(sprintf("I(%s == 2) ~ group + log_length + (1|post_id)", response_col))
    fit_fn_A <- function(f, d) glmer(f, data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  } else {
    fA_formula <- as.formula(sprintf("%s ~ group + log_length + (1|post_id)", response_col))
    fit_fn_A <- function(f, d) clmm(f, data = d, Hess = TRUE)
  }
  fA <- fit_or_load(mechanism, "A_group_omnibus", function() fit_fn_A(fA_formula, results))
  convA <- check_convergence(fA)
  cat(sprintf("  converged: %s%s\n", convA$converged, ifelse(is.na(convA$message), "", paste0(" - ", convA$message))))
  anovaA <- anova_table(fA, mechanism, "A_group_omnibus", fA_formula, "group", results, fit_fn_A)

  length_coef <- if (use_binary) fixef(fA)[["log_length"]] else coef(fA)[["log_length"]]
  length_effect <- data.frame(mechanism = mechanism, log_length_coef = length_coef)

  emA <- emmeans(fA, ~group)
  ctA <- contrast(emA, "trt.vs.ctrl", adjust = "none")
  vs_real <- tidy_contrast(ctA, use_binary) %>% mutate(mechanism = mechanism)

  # --- Model B: model * method, synthetic only ---
  cat(sprintf("--- %s: Model B (model*method, n=%d) ---\n", mechanism, nrow(synth)))
  if (use_binary) {
    fB_formula <- as.formula(sprintf("I(%s == 2) ~ model * method + log_length + (1|post_id)", response_col))
    fit_fn_B <- function(f, d) glmer(f, data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
  } else {
    fB_formula <- as.formula(sprintf("%s ~ model * method + log_length + (1|post_id)", response_col))
    fit_fn_B <- function(f, d) clmm(f, data = d, Hess = TRUE)
  }
  fB <- fit_or_load(mechanism, "B_model_x_method", function() fit_fn_B(fB_formula, synth))
  convB <- check_convergence(fB)
  cat(sprintf("  converged: %s%s\n", convB$converged, ifelse(is.na(convB$message), "", paste0(" - ", convB$message))))
  anovaB <- anova_table(fB, mechanism, "B_model_x_method", fB_formula, "model:method", synth, fit_fn_B)

  # Simple-effects contrasts (the interaction is significant everywhere, so
  # pooled main-effect means would be misleading - see report discussion):
  # between models within each method, and between methods within each model.
  em_by_method <- emmeans(fB, ~ model | method)
  ct_between_model <- contrast(em_by_method, "pairwise", adjust = "none")
  between_model <- tidy_contrast(ct_between_model, use_binary) %>% mutate(mechanism = mechanism)

  em_by_model <- emmeans(fB, ~ method | model)
  ct_within_model <- contrast(em_by_model, "pairwise", adjust = "none")
  within_model <- tidy_contrast(ct_within_model, use_binary) %>% mutate(mechanism = mechanism)

  list(
    convergence = data.frame(
      mechanism = mechanism,
      model_stage = c("A_group_omnibus", "B_model_x_method"),
      model_type = if (use_binary) c("glmer", "glmer") else c("clmm", "clmm"),
      converged = c(convA$converged, convB$converged),
      message = c(convA$message, convB$message)
    ),
    anova = bind_rows(anovaA, anovaB),
    length_effect = length_effect,
    vs_real = vs_real,
    between_model = between_model,
    within_model = within_model
  )
}

res_er <- fit_mechanism("ER", "ER_f", use_binary = FALSE)
res_ip <- fit_mechanism("IP", "IP_label", use_binary = TRUE)
res_ex <- fit_mechanism("EX", "EX_label", use_binary = TRUE)

# --- Robustness check (IP only): does dropping log_length from Model A
# change the grok/openai vs.-real picture? Separate cached fit, doesn't
# touch the A/B fits above. See report for why this was worth checking:
# openai writes much longer comments than grok, and length is a strong
# positive predictor for IP, so some of the raw group differences could be
# a length effect in disguise rather than a real model-vs-model difference.
cat("\n=== IP: Model A without log_length (robustness check) ===\n")
fA_nolen <- fit_or_load("IP", "A_nolength", function() {
  glmer(I(IP_label == 2) ~ group + (1|post_id), data = results, family = binomial,
        control = glmerControl(optimizer = "bobyqa"))
})
em_nolen <- emmeans(fA_nolen, ~group)
ct_nolen <- contrast(em_nolen, "trt.vs.ctrl", adjust = "none")
vs_real_nolength <- tidy_contrast(ct_nolen, TRUE) %>%
  mutate(mechanism = "IP", p_adj_bh = p.adjust(p_value, method = "BH"))
write.csv(vs_real_nolength, file.path(out_dir, "mixed_model_contrasts_vs_real_nolength_IP.csv"), row.names = FALSE)
print(vs_real_nolength %>% filter(grepl("grok|openai", contrast)))

all_convergence <- bind_rows(res_er$convergence, res_ip$convergence, res_ex$convergence)
all_anova <- bind_rows(res_er$anova, res_ip$anova, res_ex$anova)
all_length_effect <- bind_rows(res_er$length_effect, res_ip$length_effect, res_ex$length_effect)
all_vs_real <- bind_rows(res_er$vs_real, res_ip$vs_real, res_ex$vs_real)
all_between_model <- bind_rows(res_er$between_model, res_ip$between_model, res_ex$between_model)
all_within_model <- bind_rows(res_er$within_model, res_ip$within_model, res_ex$within_model)

# BH correction within each contrast table's own family (27 rows: 3
# mechanisms x 9 contrasts), consistent with how the Mann-Whitney table in
# 3_analyze_empathy_differences.py is corrected.
all_vs_real$p_adj_bh <- p.adjust(all_vs_real$p_value, method = "BH")
all_between_model$p_adj_bh <- p.adjust(all_between_model$p_value, method = "BH")
all_within_model$p_adj_bh <- p.adjust(all_within_model$p_value, method = "BH")

# log_length's own significance (from Model A's omnibus fit) alongside its
# coefficient, so the report can state both direction and significance.
all_length_effect <- all_length_effect %>%
  left_join(
    all_anova %>% filter(model_stage == "A_group_omnibus", term == "log_length") %>%
      select(mechanism, chisq, df, p_value),
    by = "mechanism"
  )

write.csv(all_convergence, file.path(out_dir, "mixed_model_convergence.csv"), row.names = FALSE)
write.csv(all_anova, file.path(out_dir, "mixed_model_anova.csv"), row.names = FALSE)
write.csv(all_length_effect, file.path(out_dir, "mixed_model_length_effect.csv"), row.names = FALSE)
write.csv(all_vs_real, file.path(out_dir, "mixed_model_contrasts_vs_real.csv"), row.names = FALSE)
write.csv(all_between_model, file.path(out_dir, "mixed_model_contrasts_between_model.csv"), row.names = FALSE)
write.csv(all_within_model, file.path(out_dir, "mixed_model_contrasts_within_model.csv"), row.names = FALSE)

cat("\n=== Convergence ===\n")
print(all_convergence)
cat("\n=== ANOVA (Wald chi-sq, Type II) ===\n")
print(all_anova)
cat("\n=== Contrasts vs. real (Model A) ===\n")
print(all_vs_real)
cat("\n=== Between-model, within-method contrasts (Model B) ===\n")
print(all_between_model)
cat("\n=== Within-model, between-method contrasts (Model B) ===\n")
print(all_within_model)

cat("\n=== log(length) coefficient (Model A) ===\n")
print(all_length_effect)

cat("\nWrote mixed_model_{convergence,anova,length_effect,contrasts_vs_real,contrasts_between_model,contrasts_within_model}.csv to", out_dir, "\n")
