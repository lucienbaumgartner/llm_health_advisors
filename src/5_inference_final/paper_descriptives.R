#!/usr/bin/env Rscript
#
# Descriptive statistics for the paper (paper_empirical_section.Rmd).
#
# Reads the large reply-level file (inference_results.csv, ~250 MB) ONCE and writes a few
# small CSVs (paper_desc_*.csv in output/5_inference_final). The Rmd reads only those, so
# that compiling it stays fast. Re-run this script only if inference_results.csv changes.
#
# Run from src/5_inference_final:  Rscript paper_descriptives.R
# Experts-only sensitivity run (see paper_config.R):  EXCLUDE_LAYPEOPLE=1 Rscript paper_descriptives.R

suppressMessages(library(data.table))

source("paper_config.R")
out_dir <- OUT_DIR
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
d <- fread(file.path(IN_DIR, "inference_results.csv"),
           select = c("post_id", "submission_id", "source", "model", "method", "author_type", "response_post",
                      "ER_label", "IP_label", "EX_label"))
d <- d[!is.na(response_post) & response_post != ""]

# laypeople / unverified users (EXCLUDE_LAYPEOPLE=1): count what is removed, then remove it
n_real_before <- d[source == "real", .N]
n_syn_before <- d[source == "synthetic", .N]
syn_pre <- d[source == "synthetic", .(n_syn = .N), by = post_id]
threads_real_before <- d[source == "real", unique(post_id)]
d <- drop_laypeople(d)
n_lay_excluded <- if (EXCLUDE_LAYPEOPLE) n_real_before - d[source == "real", .N] else 0L
n_threads_lost <- if (EXCLUDE_LAYPEOPLE) length(setdiff(threads_real_before, d[source == "real", unique(post_id)])) else 0L
d[, n_words := pmax(lengths(strsplit(response_post, "\\s+")), 1L)]
d[, response_post := NULL]
d[, group := ifelse(source == "real", "real", paste(model, method, sep = "_"))]

# same weights as the models (3i / 3j)
rc <- d[source == "real", .(real_count = .N), by = post_id]
d <- merge(d, rc, by = "post_id", all.x = TRUE)
d[is.na(real_count), real_count := 0L]
d[, cell_n := .N, by = .(post_id, model, method)]
d[, w := ifelse(source == "real", 1, pmin(1, real_count / cell_n))]
d <- d[w > 0]
# check: no AI reply is left in a thread without a real reply, and exactly the AI replies of the lost threads are gone
stopifnot(all(d[source == "synthetic", unique(post_id)] %in% d[source == "real", unique(post_id)]))
lost_threads <- setdiff(threads_real_before, d[source == "real", unique(post_id)])
stopifnot(n_syn_before - d[source == "synthetic", .N] == syn_pre[post_id %in% lost_threads, sum(n_syn)])
cat(sprintf("Threads without a real reply after exclusion: %d (AI replies removed with them: %d of %d)\n", length(lost_threads), syn_pre[post_id %in% lost_threads, sum(n_syn)], n_syn_before))

wm <- function(x, w) sum(w * x) / sum(w)
grp <- d[, .(n = .N, n_threads = uniqueN(post_id),
             words_mean = mean(n_words), words_median = as.numeric(median(n_words)),
             words_sd = sd(n_words), words_p90 = as.numeric(quantile(n_words, 0.9)),
             ER_raw = mean(ER_label >= 1), IP_raw = mean(IP_label >= 1), EX_raw = mean(EX_label >= 1),
             ER_w = wm(ER_label >= 1, w), IP_w = wm(IP_label >= 1, w), EX_w = wm(EX_label >= 1, w),
             weight_sum = sum(w)), by = group]
fwrite(grp, file.path(out_dir, "paper_desc_groups.csv"))

# label distributions (0/1/2) by source
lab <- rbindlist(lapply(c("ER", "IP", "EX"), function(m) {
  x <- d[, .(n = .N), by = .(source, label = get(paste0(m, "_label")))]
  x[, mechanism := m]; x
}))
fwrite(lab, file.path(out_dir, "paper_desc_labels.csv"))

# who wrote the real replies (verification flair on r/AskDocs), coarse categories + full list
auth_cat <- function(x) {
  fcase(is_layperson(x), "Layperson / not verified",
        grepl("Physician Assistant", x), "Other health professional",
        grepl("Physician|Surgeon|Neurosurg|^IM/", x), "Physician",
        grepl("Nurse|^RN|^ED RN|Midwife|Nursing", x), "Nurse / midwife",
        grepl("Medical [Ss]tudent", x), "Medical student",
        default = "Other health professional")
}
real_auth <- d[source == "real", .(n = .N), by = author_type][order(-n)]
real_auth[, category := auth_cat(author_type)]
fwrite(real_auth, file.path(out_dir, "paper_desc_authors.csv"))

# thread structure: real replies per thread, synthetic cell sizes vs real_count
th <- unique(d[, .(post_id, real_count)])
cells <- unique(d[source == "synthetic", .(post_id, model, method, cell_n, real_count)])
overall <- data.table(
  name = c("n_replies", "n_real", "n_synthetic", "n_threads", "n_threads_with_real",
          "real_per_thread_mean", "real_per_thread_median", "real_per_thread_max",
          "real_threads_single_reply", "synthetic_cells", "cells_downweighted",
          "cell_n_median", "cell_n_max", "synthetic_weight_sum", "real_weight_sum",
           "real_laypeople_excluded", "threads_lost_no_real_reply", "synthetic_replies_lost"),
  value = c(nrow(d), d[source == "real", .N], d[source == "synthetic", .N], uniqueN(d$post_id), nrow(th),
            mean(th$real_count), median(th$real_count), max(th$real_count),
            sum(th$real_count == 1), nrow(cells), sum(cells$cell_n > cells$real_count),
            median(cells$cell_n), max(cells$cell_n), d[source == "synthetic", sum(w)], d[source == "real", sum(w)],
            n_lay_excluded, n_threads_lost, n_syn_before - d[source == "synthetic", .N]))
fwrite(overall, file.path(out_dir, "paper_desc_overall.csv"))
cat("Wrote paper_desc_{groups,labels,authors,overall}.csv\n")
print(overall)

# thread-level facts from the analysis dataset (time window, post length, requested AI replies per thread)
if (requireNamespace("jsonlite", quietly = TRUE)) {
  con <- gzfile("../../input/calls_final/analysis_dataset_6600.jsonl.gz")
  th_all <- jsonlite::stream_in(con, verbose = FALSE)
  # experts-only run: describe only the threads that keep at least one real reply
  if (EXCLUDE_LAYPEOPLE) th_all <- th_all[th_all$submission_id %in% unique(d$submission_id), ]
  tt <- data.table(created = as.numeric(th_all$created_utc), n_comments = as.numeric(th_all$n_comments),
                   post_words = lengths(strsplit(as.character(th_all$selftext), "\\s+")))
  fwrite(data.table(
    name = c("threads", "window_start", "window_end", "post_words_median", "post_words_mean", "n_comments_sum"),
    value = c(nrow(tt), format(as.POSIXct(min(tt$created), origin = "1970-01-01", tz = "UTC"), "%Y-%m-%d"),
              format(as.POSIXct(max(tt$created), origin = "1970-01-01", tz = "UTC"), "%Y-%m-%d"),
              median(tt$post_words), round(mean(tt$post_words)), sum(tt$n_comments))),
    file.path(out_dir, "paper_desc_threads.csv"))
  cat("Wrote paper_desc_threads.csv\n")
}

# per-model reply length (all three strategies pooled) -- true medians over replies
fwrite(d[source == "synthetic", .(n = .N, words_median = as.numeric(median(n_words)), words_mean = mean(n_words)), by = model],
       file.path(out_dir, "paper_desc_models.csv"))
cat("Wrote paper_desc_models.csv\n")
