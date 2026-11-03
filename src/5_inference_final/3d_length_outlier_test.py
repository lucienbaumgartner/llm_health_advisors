#!/usr/bin/env python3
"""Diagnostic only - does not filter or write inference_results.csv.

Tests several candidate definitions of "comment-length outlier" against the
synthetic comments, to see how much data each would remove before settling
on one. Requested alongside a second, separate step (not implemented here):
randomly subsampling synthetic comments per thread (post_id) so each
(model, method) group has exactly as many comments per thread as the real
group does. The two interact - outlier removal happens first, so a
(post_id, model, method) cell can end up with *fewer* synthetic comments
left than that thread's real count, making an exact match impossible for
that cell. Per the agreed policy, a shortfall cell just keeps whatever
synthetic comments remain (no comments are ever invented, and no thread is
dropped for falling short). This script measures, for each candidate outlier
method, how often that shortfall actually happens and how much data it
costs on top of the balancing step's own reduction.

Length is word count via Python's `str.split()` (handles leading/trailing/
repeated whitespace correctly) - a cleaner definition than the R pipeline's
`strsplit(x, "\\s+")`, which quirkily counts a phantom extra "word" for
strings with leading whitespace. That mismatch is at most +/-1 word on a
handful of comments and irrelevant to outlier detection at this scale;
documented here rather than silently matched, since exact parity with the
older length numbers isn't the goal of a new diagnostic.

Outlier detection is done separately within each (model, method) group
(not pooled across groups), so a group that's systematically longer/shorter
than others (e.g. openai vs. grok - already known to differ) isn't
penalized for being different from other groups, only for being unusual
within its own group.
"""
import numpy as np
import pandas as pd

IN_PATH = "../../output/5_inference_final/inference_results.csv"
OUT_PATH = "../../output/5_inference_final/length_outlier_test_summary.csv"

df = pd.read_csv(IN_PATH, usecols=["post_id", "source", "model", "method", "response_post",
                                   "ER_label", "IP_label", "EX_label"])
df = df[df["response_post"].notna() & (df["response_post"] != "")].copy()

df["length"] = df["response_post"].str.split().str.len()
df["log_length"] = np.log(df["length"].clip(lower=1))

real = df[df["source"] == "real"]
synth = df[df["source"] == "synthetic"].copy()
real_count_per_post = real.groupby("post_id").size().rename("real_count")

print(f"n synthetic = {len(synth)}, n real = {len(real)}, n posts = {df['post_id'].nunique()}")
print("\nSynthetic length distribution overall (words):")
print(synth["length"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]))


def iqr_mask(x, col, k):
    q1, q3 = x[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x[col] < lo) | (x[col] > hi)


def mad_zscore_mask(x, col, threshold):
    med = x[col].median()
    mad = (x[col] - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=x.index)
    modified_z = 0.6745 * (x[col] - med) / mad
    return modified_z.abs() > threshold


def percentile_mask(x, col, lo_pct, hi_pct):
    lo, hi = x[col].quantile([lo_pct / 100, hi_pct / 100])
    return (x[col] < lo) | (x[col] > hi)


METHODS = {
    "iqr_1.5_raw": lambda g: iqr_mask(g, "length", 1.5),
    "iqr_1.5_log": lambda g: iqr_mask(g, "log_length", 1.5),
    "iqr_3.0_log": lambda g: iqr_mask(g, "log_length", 3.0),
    "mad_3.5_log": lambda g: mad_zscore_mask(g, "log_length", 3.5),
    "percentile_1_99_log": lambda g: percentile_mask(g, "log_length", 1, 99),
    "percentile_2.5_97.5_log": lambda g: percentile_mask(g, "log_length", 2.5, 97.5),
}

results = []
for name, fn in METHODS.items():
    outlier_mask = synth.groupby(["model", "method"], group_keys=False).apply(fn, include_groups=False)
    n_excluded = int(outlier_mask.sum())

    remaining = synth.loc[~outlier_mask]
    remaining_counts = remaining.groupby(["post_id", "model", "method"]).size().rename("remaining").reset_index()
    original_counts = synth.groupby(["post_id", "model", "method"]).size().rename("original").reset_index()

    cells = original_counts.merge(remaining_counts, on=["post_id", "model", "method"], how="left")
    cells["remaining"] = cells["remaining"].fillna(0).astype(int)
    cells = cells.merge(real_count_per_post, on="post_id", how="left")

    # Balanced total under "keep whatever remains" for the OUTLIER-FILTERED
    # data, vs. the balanced total you'd get from the UNFILTERED data (i.e.
    # balancing alone, no outlier removal) - the gap between the two is the
    # marginal cost of this specific outlier method, isolated from the
    # balancing step's own inherent reduction (matching each group down to
    # real_count per thread already removes data regardless of outliers).
    cells["matched_after_outliers"] = np.minimum(cells["remaining"], cells["real_count"])
    cells["matched_no_outliers"] = np.minimum(cells["original"], cells["real_count"])
    shortfall_cells = cells[cells["remaining"] < cells["real_count"]]

    results.append({
        "method": name,
        "synthetic_excluded_n": n_excluded,
        "synthetic_excluded_pct": round(100 * n_excluded / len(synth), 3),
        "cells_total": len(cells),
        "shortfall_cells_n": len(shortfall_cells),
        "shortfall_cells_pct": round(100 * len(shortfall_cells) / len(cells), 3),
        "shortfall_comments_total": int((cells["real_count"] - cells["remaining"]).clip(lower=0).sum()),
        "balanced_total_with_outlier_removal": int(cells["matched_after_outliers"].sum()),
        "balanced_total_without_outlier_removal": int(cells["matched_no_outliers"].sum()),
        "marginal_loss_from_outlier_removal": int(
            cells["matched_no_outliers"].sum() - cells["matched_after_outliers"].sum()
        ),
    })

summary = pd.DataFrame(results)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
print("\n=== Outlier method comparison ===")
print(summary.to_string(index=False))

# Reference: what balancing alone (no outlier removal at all) already costs,
# so the marginal costs above have a baseline to compare against.
no_outlier_cells = synth.groupby(["post_id", "model", "method"]).size().rename("original").reset_index()
no_outlier_cells = no_outlier_cells.merge(real_count_per_post, on="post_id", how="left")
baseline_matched = int(np.minimum(no_outlier_cells["original"], no_outlier_cells["real_count"]).sum())
print(f"\nBaseline (balancing only, no outlier removal): {baseline_matched} synthetic comments "
      f"retained per group after per-thread matching (real total = {len(real)}, "
      f"synthetic total before any of this = {len(synth)}).")

summary.to_csv(OUT_PATH, index=False)
print(f"\nWrote comparison table to {OUT_PATH}")

# ---------------------------------------------------------------------------
# Per-(model, method) breakdowns for every rule, plus a no-exclusion baseline
# ("none"), for report_outlier_exclusion.Rmd:
#   (i)  share of (post_id, model, method) cells that end up with fewer
#        synthetic comments than the thread's real count ("shortfall"), and
#   (ii) share of each group's non-zero-rated synthetic comments that the rule
#        would exclude, per mechanism.
# The baseline shows shortfall that already exists without any exclusion (a few
# cells have fewer synthetic comments than real ones to begin with), so the
# rules' added shortfall isn't over-attributed to them. Column `outlier_rule`
# is the exclusion rule; `method` is the prompting method (dg/sag/sagii).
# ---------------------------------------------------------------------------
SHORTFALL_OUT = "../../output/5_inference_final/length_outlier_test_shortfall_by_group.csv"
NONZERO_OUT = "../../output/5_inference_final/length_outlier_test_nonzero_by_group.csv"

rules = {"none": lambda g: pd.Series(False, index=g.index), **METHODS}
shortfall_rows, nonzero_rows = [], []
for rule_name, fn in rules.items():
    mask = synth.groupby(["model", "method"], group_keys=False).apply(fn, include_groups=False)
    synth["is_outlier"] = mask.reindex(synth.index).fillna(False).astype(bool)

    remaining = synth.loc[~synth["is_outlier"]].groupby(["post_id", "model", "method"]).size().rename("remaining").reset_index()
    original = synth.groupby(["post_id", "model", "method"]).size().rename("original").reset_index()
    cells = original.merge(remaining, on=["post_id", "model", "method"], how="left")
    cells["remaining"] = cells["remaining"].fillna(0).astype(int)
    cells = cells.merge(real_count_per_post, on="post_id", how="left")
    cells["short"] = cells["remaining"] < cells["real_count"]
    g = cells.groupby(["model", "method"]).agg(n_cells=("short", "size"), n_shortfall_cells=("short", "sum")).reset_index()
    g["pct_shortfall_cells"] = 100 * g["n_shortfall_cells"] / g["n_cells"]
    g.insert(0, "outlier_rule", rule_name)
    shortfall_rows.append(g)

    for mech in ["ER", "IP", "EX"]:
        nz = synth[f"{mech}_label"] != 0
        t = pd.DataFrame({"model": synth["model"], "method": synth["method"], "nz": nz,
                          "nz_excl": nz & synth["is_outlier"]})
        a = t.groupby(["model", "method"]).agg(n_nonzero=("nz", "sum"), n_nonzero_excluded=("nz_excl", "sum")).reset_index()
        a["pct_nonzero_excluded"] = 100 * a["n_nonzero_excluded"] / a["n_nonzero"]
        a.insert(0, "mechanism", mech)
        a.insert(0, "outlier_rule", rule_name)
        nonzero_rows.append(a)

pd.concat(shortfall_rows).to_csv(SHORTFALL_OUT, index=False)
pd.concat(nonzero_rows).to_csv(NONZERO_OUT, index=False)
print(f"Wrote {SHORTFALL_OUT}\nWrote {NONZERO_OUT}")
