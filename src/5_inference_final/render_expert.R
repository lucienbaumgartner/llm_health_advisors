# Renders the experts-only version of the paper (EXCLUDE_LAYPEOPLE=1).
# The abstract in the YAML header is static text with the numbers of the main run; here they are replaced by the
# numbers of the experts-only run in a temporary copy of the Rmd, which is deleted afterwards.
Sys.setenv(EXCLUDE_LAYPEOPLE = "1")
source("paper_config.R")
rd <- function(f) read.csv(file.path(OUT_DIR, f), stringsAsFactors = FALSE)
f2  <- function(x) ifelse(abs(x) < 0.1, sprintf("%.3f", x), sprintf("%.2f", x))
fmt <- function(x) format(round(x), big.mark = ",", trim = TRUE)
ov  <- setNames(rd("paper_desc_overall.csv")$value, rd("paper_desc_overall.csv")$name)
tp  <- rd("v2_topic_gap_by_topic.csv"); avg <- tp[is.na(tp$topic) | tp$topic == "", ]
ge  <- rd("v3_group_effects.csv"); gm <- function(w) ge$or[ge$spec == "length adjusted" & ge$level == "model" & ge$mechanism == "IP" & ge$who == w]
x <- readLines("paper_empirical_section.Rmd", encoding = "UTF-8")
end <- which(x == "keywords:")[1]                                  # abstract = lines before this
rep <- c("8,625" = fmt(ov[["n_real"]]), "6,600" = fmt(ov[["n_threads"]]), "128,653" = fmt(ov[["n_synthetic"]]),
         "0.087" = f2(avg$gap_or[avg$mechanism == "ER"]), "0.055" = f2(avg$gap_or[avg$mechanism == "EX"]),
         "1.97" = f2(gm("grok")), "0.44" = f2(gm("gemini")), "0.74" = f2(gm("openai")))
for (k in names(rep)) { i <- grep(k, x[1:end], fixed = TRUE); stopifnot(length(i) == 1); x[i] <- sub(k, rep[[k]], x[i], fixed = TRUE) }
tmp <- ".paper_expert_tmp.Rmd"; writeLines(x, tmp, useBytes = TRUE)
print(rep)
rmarkdown::render(tmp, output_file = "paper_empirical_section_expert.pdf", quiet = TRUE); unlink(".paper_expert_tmp.Rmd")
