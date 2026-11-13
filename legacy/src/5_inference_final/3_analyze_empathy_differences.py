"""
Compare predicted empathy (Emotional Reactions / Interpretations / Explorations)
between real clinician comments and synthetic comments, broken down by
synthetic model (gemini, openai, grok) and prompting method (dg, sag, sagii).

Empathy labels are ordinal (0 = none, 1 = weak, 2 = strong). This script
computes:
  - Mann-Whitney U for each (model, method) group vs. real, per mechanism
    (pairwise, 27 tests total), with Benjamini-Hochberg FDR correction and
    rank-biserial correlation as effect size.
  - Whether comment length (word count) correlates with predicted empathy
    (Spearman), overall and within each group.

The omnibus (Kruskal-Wallis) and by-method/by-model pooled comparisons that
used to live here have been replaced by proper mixed-effects models (which
account for comments clustering by post, and control for comment length
directly) - see 3b_mixed_effects_models.R. The Mann-Whitney table above is
kept as-is: it's simpler/more transparent for the fine-grained real-vs-each-
group comparison, and doesn't need a clustering correction the same way an
omnibus/interaction test does.

Unlike pilot 1, no posts are dropped here: real and synthetic comment counts
differ by design (synthetic comments are generated 1:1 against each thread's
total comment count, not against the smaller real_comments subset - see
1_prepare_data.py), not because of a data-quality bug.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

MECHANISMS = ["ER", "IP", "EX"]
MECHANISM_NAMES = {
    "ER": "Emotional Reactions",
    "IP": "Interpretations",
    "EX": "Explorations",
}
MODELS = ["gemini", "openai", "grok"]
METHODS = ["dg", "sag", "sagii"]
SYNTH_GROUPS = [f"{model}_{method}" for model in MODELS for method in METHODS]


def benjamini_hochberg(pvals):
    """NaN-safe BH correction: NaN p-values (e.g. from a degenerate/constant
    input to a correlation test) are excluded from the correction and stay
    NaN in the output, rather than propagating into every other entry via
    the cumulative-min step."""
    pvals = np.asarray(pvals, dtype=float)
    adjusted = np.full(pvals.shape, np.nan)
    valid = ~np.isnan(pvals)
    valid_pvals = pvals[valid]
    n = len(valid_pvals)
    if n == 0:
        return adjusted
    order = np.argsort(valid_pvals)
    ranked = valid_pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    valid_adjusted = np.empty(n)
    valid_adjusted[order] = np.clip(ranked, 0, 1)
    adjusted[valid] = valid_adjusted
    return adjusted


def rank_biserial(x, y, u_stat):
    # r = 1 - 2U / (n1 * n2), U from mannwhitneyu with x as the first sample
    n1, n2 = len(x), len(y)
    return 1 - (2 * u_stat) / (n1 * n2)


def summarize_group(labels):
    labels = np.asarray(labels)
    n = len(labels)
    counts = {f"prop_label_{k}": float((labels == k).mean()) if n else np.nan for k in (0, 1, 2)}
    return {
        "n": n,
        "mean": float(labels.mean()) if n else np.nan,
        **counts,
    }


def run_analysis(df, out_dir):
    """Compute all summary/comparison/correlation tables for `df` and write
    them to out_dir. Returns the DataFrames (for console printing) as a dict."""
    summary_rows = []
    comparison_rows = []

    for mech in MECHANISMS:
        col = f"{mech}_label"
        real_labels = df.loc[df["group"] == "real", col].values

        for group in ["real"] + SYNTH_GROUPS:
            labels = df.loc[df["group"] == group, col].values
            summary_rows.append({"mechanism": mech, "group": group, **summarize_group(labels)})

        for group in SYNTH_GROUPS:
            model, method = group.split("_", 1)
            synth_labels = df.loc[df["group"] == group, col].values
            u_stat, p_val = mannwhitneyu(synth_labels, real_labels, alternative="two-sided")
            effect_r = rank_biserial(synth_labels, real_labels, u_stat)
            comparison_rows.append({
                "mechanism": mech,
                "model": model,
                "method": method,
                "n_real": len(real_labels),
                "n_synthetic": len(synth_labels),
                "mean_real": float(real_labels.mean()),
                "mean_synthetic": float(synth_labels.mean()),
                "U": u_stat,
                "p_value": p_val,
                "rank_biserial_r": effect_r,
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["p_adj_bh"] = benjamini_hochberg(comparison_df["p_value"].values)
    comparison_df["significant_bh_0.05"] = comparison_df["p_adj_bh"] < 0.05

    summary_df = pd.DataFrame(summary_rows)

    # --- comment length vs. predicted empathy ---
    length_rows = []
    for mech in MECHANISMS:
        col = f"{mech}_label"
        for group in ["overall", "real"] + SYNTH_GROUPS:
            sub = df if group == "overall" else df.loc[df["group"] == group]
            rho, p_val = spearmanr(sub["length"], sub[col])
            length_rows.append({
                "mechanism": mech,
                "group": group,
                "n": len(sub),
                "spearman_rho": rho,
                "p_value": p_val,
            })
    length_df = pd.DataFrame(length_rows)
    length_df["p_adj_bh"] = benjamini_hochberg(length_df["p_value"].values)
    length_df["significant_bh_0.05"] = length_df["p_adj_bh"] < 0.05

    length_summary_df = (
        df.groupby("group")["length"]
        .agg(n="count", mean_length="mean", median_length="median", std_length="std")
        .reindex(["real"] + SYNTH_GROUPS)
        .reset_index()
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "empathy_summary_stats.csv": summary_df,
        "empathy_group_comparison.csv": comparison_df,
        "empathy_length_correlation.csv": length_df,
        "comment_length_by_group.csv": length_summary_df,
    }
    for filename, out_df in tables.items():
        out_path = out_dir / filename
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    print("\n=== Summary (mean predicted empathy label, 0-2) ===")
    print(summary_df.pivot(index="group", columns="mechanism", values="mean").loc[["real"] + SYNTH_GROUPS])

    print("\n=== Real vs. each (model, method) group (Mann-Whitney U, BH-adjusted) ===")
    print(comparison_df[["mechanism", "model", "method", "mean_real", "mean_synthetic",
                          "p_value", "p_adj_bh", "rank_biserial_r"]].to_string(index=False))

    print("\n=== Comment length (word count) by group ===")
    print(length_summary_df.to_string(index=False))

    print("\n=== Comment length vs. predicted empathy (Spearman, BH-adjusted) ===")
    print(length_df.to_string(index=False))

    return {
        "summary": summary_df,
        "comparison": comparison_df,
        "length": length_df,
        "length_summary": length_summary_df,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, default="../../output/5_inference_final/inference_results.csv")
    parser.add_argument("--output_dir", type=str, default="../../output/5_inference_final")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    n_missing = df["response_post"].isna().sum()
    if n_missing:
        print(f"Dropping {n_missing} comments with an empty/missing response_post "
              f"(a handful of synthetic generations came back blank - see qc_comment_counts.csv).")
        df = df[df["response_post"].notna()].copy()

    df["group"] = df["source"].where(df["source"] == "real", df["model"] + "_" + df["method"])
    df["length"] = df["response_post"].astype(str).str.split().str.len()

    print(f"\n########## All comments (n={len(df)}) ##########")
    run_analysis(df, args.output_dir)


if __name__ == "__main__":
    main()
