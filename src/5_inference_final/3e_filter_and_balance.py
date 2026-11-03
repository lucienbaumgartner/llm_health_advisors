#!/usr/bin/env python3
"""Applies the two data-cleaning steps agreed on after 3d_length_outlier_test.py's
comparison, in this specific order (the order matters - see below):

1. Length-outlier removal (synthetic only): MAD-based modified z-score on
   log(length), threshold 3.5, computed separately within each (model,
   method) group - settled on after comparing several candidate definitions
   (see 3d and length_outlier_test_summary.csv). Removes 332 comments
   (0.26% of synthetic), causes 105 (post_id, model, method) cells to fall
   short of their thread's real count, and empties 82 cells to zero
   entirely (81 of those started with exactly 1 synthetic comment). Both the
   excluded comments and the real comments on the affected threads are
   overwhelmingly empathy-0 across all three mechanisms (79-95% zero
   depending on mechanism/side) - discussed and accepted as an acceptable
   cost. The exclusion rate is NOT evenly spread across (model, method)
   groups (chi2=184.83, df=8, p=1e-35): concentrated in Gemini/Grok's
   SAG/SAGII outputs, negligible for OpenAI.

2. Per-thread balancing (synthetic only): for each (post_id, model, method)
   cell, randomly subsample down to exactly that thread's real comment
   count, so every group is compared against real using the same per-thread
   sample size instead of synthetic being systematically overrepresented
   (synthetic was generated 1:1 against each thread's *total* comment count,
   real_comments is a smaller pre-filtered subset - see qc_comment_counts.csv
   and this folder's README).

Order matters because the two interact: outlier removal happens FIRST, so a
cell can end up with fewer synthetic comments remaining than its thread's
real count. Per the agreed policy, that cell just keeps whatever synthetic
comments remain rather than being dropped or backfilled - no thread is ever
removed and no comment is ever invented. Doing it in the other order
(balance first, then remove outliers from the already-downsampled set) was
not what was tested/decided on and would remove a different, smaller set of
comments (outlier detection on a smaller, already-subsampled pool behaves
differently).

Real comments are never touched by either step - only synthetic comments are
filtered/subsampled.

Subsampling uses a fixed seed (matching this project's established
reproducibility standard - see 3c_bayesian_models.R) so re-running this
script reproduces the exact same selection.
"""
import numpy as np
import pandas as pd

IN_PATH = "../../output/5_inference_final/inference_results.csv"
OUT_PATH = "../../output/5_inference_final/inference_results_filtered_balanced.csv"
QC_OUT_PATH = "../../output/5_inference_final/qc_filter_and_balance.csv"
QC_BY_GROUP_PATH = "../../output/5_inference_final/qc_outlier_by_group.csv"
QC_EMPATHY_PATH = "../../output/5_inference_final/qc_emptied_cell_empathy.csv"

MAD_THRESHOLD = 3.5
SEED = 20260101

df = pd.read_csv(IN_PATH)
df = df[df["response_post"].notna() & (df["response_post"] != "")].copy()
df["length"] = df["response_post"].str.split().str.len()
df["log_length"] = np.log(df["length"].clip(lower=1))

real = df[df["source"] == "real"].copy()
synth = df[df["source"] == "synthetic"].copy()
real_count_per_post = real.groupby("post_id").size().rename("real_count")

print(f"Starting: {len(real)} real, {len(synth)} synthetic comments, {df['post_id'].nunique()} posts.")

# --- Step 1: MAD 3.5x length-outlier removal, per (model, method) group ---


def mad_zscore_mask(x, col="log_length", threshold=MAD_THRESHOLD):
    med = x[col].median()
    mad = (x[col] - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=x.index)
    modified_z = 0.6745 * (x[col] - med) / mad
    return modified_z.abs() > threshold


outlier_mask = synth.groupby(["model", "method"], group_keys=False).apply(mad_zscore_mask, include_groups=False)
synth["is_outlier"] = outlier_mask
synth_kept = synth.loc[~outlier_mask].copy()
print(f"After outlier removal: {len(synth_kept)} synthetic comments "
      f"({outlier_mask.sum()} excluded, {100 * outlier_mask.sum() / len(synth):.3f}%).")

# --- Step 2: per-thread balancing, per (model, method) cell ---

synth_kept = synth_kept.merge(real_count_per_post, on="post_id", how="left")
synth_kept["real_count"] = synth_kept["real_count"].fillna(0).astype(int)

rng = np.random.default_rng(SEED)


def select_indices(group):
    target = group["real_count"].iloc[0]
    if len(group) <= target:
        return group.index.values
    return rng.choice(group.index.values, size=target, replace=False)


selected = synth_kept.groupby(["post_id", "model", "method"]).apply(select_indices, include_groups=False)
selected_idx = np.concatenate(selected.values) if len(selected) else np.array([], dtype=int)
synth_balanced = synth_kept.loc[selected_idx].drop(columns=["real_count"])
print(f"After per-thread balancing: {len(synth_balanced)} synthetic comments "
      f"(target was {int(real_count_per_post.reindex(synth_kept['post_id'].unique()).sum())}-ish per group; "
      f"real total is {len(real)}).")

# --- QC summary: before/after counts per (post_id, model, method) cell ---

original_counts = synth.groupby(["post_id", "model", "method"]).size().rename("original_n").reset_index()
kept_counts = synth_kept.groupby(["post_id", "model", "method"]).size().rename("after_outlier_removal_n").reset_index()
balanced_counts = synth_balanced.groupby(["post_id", "model", "method"]).size().rename("final_n").reset_index()

qc = original_counts.merge(kept_counts, on=["post_id", "model", "method"], how="left")
qc = qc.merge(balanced_counts, on=["post_id", "model", "method"], how="left")
qc["after_outlier_removal_n"] = qc["after_outlier_removal_n"].fillna(0).astype(int)
qc["final_n"] = qc["final_n"].fillna(0).astype(int)
qc = qc.merge(real_count_per_post, on="post_id", how="left")
qc["real_count"] = qc["real_count"].fillna(0).astype(int)
qc["shortfall_vs_real"] = (qc["real_count"] - qc["final_n"]).clip(lower=0)
qc.to_csv(QC_OUT_PATH, index=False)

# --- QC: chosen-method (MAD 3.5x) exclusion/shortfall breakdown by (model, method) ---
# ("emptied" below means a cell that HAD synthetic comments before outlier removal
# (original_n > 0) but has none left after it (final_n == 0) - distinct from the small,
# pre-existing number of cells that already had 0 synthetic comments for unrelated
# reasons (see qc_comment_counts.csv), which aren't counted here.)

by_group = synth.groupby(["model", "method"]).agg(n_synth=("is_outlier", "size"), n_excluded=("is_outlier", "sum")).reset_index()
by_group["pct_excluded"] = 100 * by_group["n_excluded"] / by_group["n_synth"]

emptied_mask = (qc["original_n"] > 0) & (qc["final_n"] == 0)
cell_stats = qc.groupby(["model", "method"]).agg(
    n_cells=("final_n", "size"),
    n_shortfall_cells=("shortfall_vs_real", lambda s: (s > 0).sum()),
).reset_index()
emptied_by_group = qc.loc[emptied_mask].groupby(["model", "method"]).size().rename("n_emptied_cells")
cell_stats = cell_stats.merge(emptied_by_group, on=["model", "method"], how="left")
cell_stats["n_emptied_cells"] = cell_stats["n_emptied_cells"].fillna(0).astype(int)
cell_stats["pct_shortfall_cells"] = 100 * cell_stats["n_shortfall_cells"] / cell_stats["n_cells"]

by_group = by_group.merge(cell_stats, on=["model", "method"])
by_group.to_csv(QC_BY_GROUP_PATH, index=False)

# --- QC: empathy label (zero vs. non-zero) for excluded comments and real comments
# on the threads where a cell got emptied entirely - is the lost data "boring"
# (empathy 0) or does it carry real signal? ---

emptied_cells_df = qc.loc[emptied_mask, ["post_id", "model", "method"]]
excluded_in_emptied = synth.loc[synth["is_outlier"]].merge(emptied_cells_df, on=["post_id", "model", "method"], how="inner")
real_on_emptied_threads = real.merge(emptied_cells_df[["post_id"]].drop_duplicates(), on="post_id", how="inner")

empathy_rows = []
for side_name, side_df in [("excluded_synthetic", excluded_in_emptied), ("real_on_affected_threads", real_on_emptied_threads)]:
    for mech in ["ER_label", "IP_label", "EX_label"]:
        n_zero = int((side_df[mech] == 0).sum())
        n_nonzero = int((side_df[mech] != 0).sum())
        empathy_rows.append({
            "side": side_name, "mechanism": mech.replace("_label", ""),
            "n_zero": n_zero, "n_nonzero": n_nonzero, "n_total": n_zero + n_nonzero,
            "pct_nonzero": round(100 * n_nonzero / (n_zero + n_nonzero), 1) if (n_zero + n_nonzero) > 0 else float("nan"),
        })
pd.DataFrame(empathy_rows).to_csv(QC_EMPATHY_PATH, index=False)

# --- Final output: real (unchanged) + balanced synthetic ---

out_cols = [c for c in df.columns if c not in ("length", "log_length")]
final = pd.concat([real[out_cols], synth_balanced[out_cols]], ignore_index=True)
final.to_csv(OUT_PATH, index=False)

print(f"\nFinal dataset: {len(final)} comments ({len(real)} real, {len(synth_balanced)} synthetic).")
print(f"Cells with a shortfall vs. real_count in the final data: "
      f"{(qc['shortfall_vs_real'] > 0).sum()} / {len(qc)}")
print(f"Cells with zero synthetic comments in the final data: {(qc['final_n'] == 0).sum()} / {len(qc)}")
print(f"\nWrote {OUT_PATH}")
print(f"Wrote {QC_OUT_PATH}")
print(f"Wrote {QC_BY_GROUP_PATH}")
print(f"Wrote {QC_EMPATHY_PATH}")
