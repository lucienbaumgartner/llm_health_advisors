"""
Compare predicted empathy (Emotional Reactions / Interpretations / Explorations)
between real clinician comments and synthetic comments, broken down by
prompting method (DG, SAG, SAGII).

Empathy labels are ordinal (0 = none, 1 = weak, 2 = strong), so group
comparisons use nonparametric tests:
  - Kruskal-Wallis across {real, DG, SAG, SAGII} per mechanism (omnibus).
  - Mann-Whitney U for each synthetic method vs. real, per mechanism
    (pairwise), with Benjamini-Hochberg FDR correction across the 9 tests
    (3 mechanisms x 3 methods).
  - Rank-biserial correlation as effect size for each pairwise test.

Also checks whether comment length (word count) correlates with predicted
empathy (Spearman, since labels are ordinal), both overall and within each
group - real and synthetic comments may differ systematically in length,
which could confound the real-vs-synthetic empathy comparisons above.

Restricted to the sheets where real and synthetic comment counts actually
match (see qc_comment_counts.csv / 1_xlsx_to_json.py) - the other sheets have
real comments merged into fewer cells than their stated count, which would
bias the real-comment group.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

MECHANISMS = ["ER", "IP", "EX"]
MECHANISM_NAMES = {
    "ER": "Emotional Reactions",
    "IP": "Interpretations",
    "EX": "Explorations",
}
METHODS = ["DG", "SAG", "SAGII"]


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.clip(ranked, 0, 1)
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
    kruskal_rows = []
    comparison_rows = []

    for mech in MECHANISMS:
        col = f"{mech}_label"
        real_labels = df.loc[df["group"] == "real", col].values

        for group in ["real"] + METHODS:
            labels = df.loc[df["group"] == group, col].values
            summary_rows.append({"mechanism": mech, "group": group, **summarize_group(labels)})

        group_arrays = [df.loc[df["group"] == g, col].values for g in ["real"] + METHODS]
        h_stat, p_kw = kruskal(*group_arrays)
        kruskal_rows.append({"mechanism": mech, "H": h_stat, "p_value": p_kw})

        for method in METHODS:
            synth_labels = df.loc[df["group"] == method, col].values
            u_stat, p_val = mannwhitneyu(synth_labels, real_labels, alternative="two-sided")
            effect_r = rank_biserial(synth_labels, real_labels, u_stat)
            comparison_rows.append({
                "mechanism": mech,
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
    kruskal_df = pd.DataFrame(kruskal_rows)

    # --- comment length vs. predicted empathy ---
    length_rows = []
    for mech in MECHANISMS:
        col = f"{mech}_label"
        for group in ["overall", "real"] + METHODS:
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
        .reindex(["real"] + METHODS)
        .reset_index()
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "empathy_summary_stats.csv": summary_df,
        "empathy_group_comparison.csv": comparison_df,
        "empathy_kruskal.csv": kruskal_df,
        "empathy_length_correlation.csv": length_df,
        "comment_length_by_group.csv": length_summary_df,
    }
    for filename, out_df in tables.items():
        out_path = out_dir / filename
        out_df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

    print("\n=== Summary (mean predicted empathy label, 0-2) ===")
    print(summary_df.pivot(index="group", columns="mechanism", values="mean").loc[["real"] + METHODS])

    print("\n=== Kruskal-Wallis (real vs. DG vs. SAG vs. SAGII) ===")
    print(kruskal_df.to_string(index=False))

    print("\n=== Real vs. each synthetic method (Mann-Whitney U, BH-adjusted) ===")
    print(comparison_df[["mechanism", "method", "mean_real", "mean_synthetic",
                          "p_value", "p_adj_bh", "rank_biserial_r"]].to_string(index=False))

    print("\n=== Comment length (word count) by group ===")
    print(length_summary_df.to_string(index=False))

    print("\n=== Comment length vs. predicted empathy (Spearman, BH-adjusted) ===")
    print(length_df.to_string(index=False))

    return {
        "summary": summary_df,
        "comparison": comparison_df,
        "kruskal": kruskal_df,
        "length": length_df,
        "length_summary": length_summary_df,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=str, default="../../output/3_inference/inference_results.csv")
    parser.add_argument("--qc_csv", type=str, default="../../output/3_inference/qc_comment_counts.csv")
    parser.add_argument("--output_dir", type=str, default="../../output/3_inference")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["group"] = df["source"].where(df["source"] == "real", df["method"])
    df["length"] = df["response_post"].astype(str).str.split().str.len()

    qc = pd.read_csv(args.qc_csv)
    matched_sheets = set(qc.loc[qc["match"], "sheet_name"])
    df_matched = df[df["sheet_name"].isin(matched_sheets)]

    print(f"\n########## Matched-count sheets only "
          f"({len(matched_sheets)}/{len(qc)} sheets, n={len(df_matched)} comments) ##########")
    run_analysis(df_matched, args.output_dir)


if __name__ == "__main__":
    main()
