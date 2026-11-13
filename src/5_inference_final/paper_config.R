# Shared settings of the paper pipeline (sourced by 3i, 3j, paper_descriptives.R, paper_joint_tests.R and the Rmd).
#
# EXCLUDE_LAYPEOPLE=1 re-runs everything with the real replies of users flagged as laypeople or not
# verified (r/AskDocs flair "Layperson/not verified as healthcare professional" or "This user has not
# yet been verified") removed. Threads that then have no real reply drop out together with their AI
# replies (the response weights are recomputed from the remaining real replies). Results go to a
# separate folder, so the main results are never overwritten:
#   default                  -> ../../output/5_inference_final         (all real replies)
#   EXCLUDE_LAYPEOPLE=1      -> ../../output/5_inference_final_expert  (verified health professionals only)
# The classifier output (inference_results.csv) is always read from the default folder.
#
# Run all scripts from src/5_inference_final; see README.md for the order.

EXCLUDE_LAYPEOPLE <- Sys.getenv("EXCLUDE_LAYPEOPLE", "0") %in% c("1", "TRUE", "true")
IN_DIR  <- "../../output/5_inference_final"
OUT_DIR <- if (EXCLUDE_LAYPEOPLE) "../../output/5_inference_final_expert" else IN_DIR
LAY_REGEX <- "Layperson|not yet been verified"

# a reply counts as lay/unverified if its flair says so or if it has no flair at all (1 reply; on r/AskDocs a missing flair means not verified)
is_layperson <- function(author_type) is.na(author_type) | trimws(author_type) == "" | grepl(LAY_REGEX, author_type, ignore.case = TRUE)

# remove the real replies of laypeople / unverified users (no-op unless EXCLUDE_LAYPEOPLE=1)
drop_laypeople <- function(d) {
  if (!EXCLUDE_LAYPEOPLE) return(d)
  d <- d[!(d$source == "real" & is_layperson(d$author_type)), ]
  # safeguard: nothing lay/unverified may be left among the real replies
  stopifnot(!any(is_layperson(d$author_type[d$source == "real"])),
            !any(grepl("layperson|verified|unverified", d$author_type[d$source == "real"], ignore.case = TRUE)))
  d
}
if (EXCLUDE_LAYPEOPLE) cat("EXCLUDE_LAYPEOPLE=1: real replies of laypeople / unverified users are removed; output to", OUT_DIR, "\n")
