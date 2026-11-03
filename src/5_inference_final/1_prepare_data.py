"""
Load the pilot-2 analysis dataset (one JSON object per Reddit submission,
gzipped JSONL) into the same posts.json shape 2_run_inference.py expects, and
check comment counts for transparency.

Each input line has:
    submission_id, title, selftext, n_comments, permalink, created_utc,
    real_comments: [{body, author_flair_text, score}, ...],
    synthetic: {model: {method: [{body, author_type}, ...]}}
      - model in {gemini, openai, grok}, method in {dg, sag, sagii}

Unlike pilot 1 (an xlsx with a real-comment-merging bug), there is no known
data-quality issue here to filter around. real_comments and each synthetic
(model, method) list often have different lengths per post - but that's by
design, not corruption: synthetic comment count matches `n_comments` (the
original thread's total comment count) in effectively 100% of cases, while
`real_comments` is a smaller, already-filtered subset of that thread (e.g.
top-level, non-removed comments - see ../1_preprocessing). So real and
synthetic counts aren't expected to match, and nothing is dropped here - the
QC report below is descriptive, not a filter.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

MODELS = ["gemini", "openai", "grok"]
METHODS = ["dg", "sag", "sagii"]


def open_maybe_gzip(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_line(obj, post_id):
    real_comments = [
        {"author_type": c.get("author_flair_text"), "comment": c.get("body"), "score": c.get("score")}
        for c in obj.get("real_comments", [])
    ]
    synthetic_comments = {
        model: {
            method: [
                {"author_type": c.get("author_type"), "comment": c.get("body")}
                for c in obj.get("synthetic", {}).get(model, {}).get(method, [])
            ]
            for method in METHODS
        }
        for model in MODELS
    }
    return {
        "post_id": post_id,
        "submission_id": obj["submission_id"],
        "title": obj.get("title"),
        "body": obj.get("selftext"),
        "n_comments": obj.get("n_comments"),
        "real_comments": real_comments,
        "synthetic_comments": synthetic_comments,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=str, default="../../input/calls_final/analysis_dataset_6600.jsonl.gz")
    parser.add_argument("--output_json", type=str, default="../../output/5_inference_final/posts.json")
    parser.add_argument("--output_qc_csv", type=str, default="../../output/5_inference_final/qc_comment_counts.csv")
    args = parser.parse_args()

    posts = []
    qc_rows = []
    n_matches_n_comments = 0
    n_total_combos = 0

    with open_maybe_gzip(args.input_path) as f:
        for post_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            post = parse_line(obj, post_id)
            posts.append(post)

            real_n = len(post["real_comments"])
            n_comments = post["n_comments"]
            for model in MODELS:
                for method in METHODS:
                    synth_n = len(post["synthetic_comments"][model][method])
                    n_total_combos += 1
                    matches_n_comments = synth_n == n_comments
                    if matches_n_comments:
                        n_matches_n_comments += 1
                    qc_rows.append({
                        "post_id": post_id,
                        "submission_id": post["submission_id"],
                        "model": model,
                        "method": method,
                        "n_comments": n_comments,
                        "real_count": real_n,
                        "synthetic_count": synth_n,
                        "synthetic_matches_n_comments": matches_n_comments,
                        "synthetic_matches_real_count": synth_n == real_n,
                    })

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

    out_csv = Path(args.output_qc_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(qc_rows)

    n_real_match = sum(r["synthetic_matches_real_count"] for r in qc_rows)
    print(f"Parsed {len(posts)} posts -> {out_json}")
    print(f"QC report -> {out_csv}")
    print(f"synthetic_count == n_comments: {n_matches_n_comments}/{n_total_combos} "
          f"({n_matches_n_comments / n_total_combos:.1%}) of (post, model, method) combos")
    print(f"synthetic_count == real_count: {n_real_match}/{n_total_combos} "
          f"({n_real_match / n_total_combos:.1%}) of (post, model, method) combos")
    print("(synthetic comments were generated 1:1 against the thread's total comment count, "
          "not against real_comments, which is a smaller filtered subset - see module docstring. "
          "Nothing is dropped for inference.)")

    total_real = sum(len(p["real_comments"]) for p in posts)
    total_synth = sum(
        len(p["synthetic_comments"][m][me])
        for p in posts for m in MODELS for me in METHODS
    )
    print(f"\nTotal comments for inference: {total_real} real, {total_synth} synthetic "
          f"({total_real + total_synth} total)")


if __name__ == "__main__":
    main()
