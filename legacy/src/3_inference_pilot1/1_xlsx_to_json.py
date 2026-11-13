"""
Convert the clinician/synthetic-comment xlsx (one sheet per Reddit post) into a
single structured JSON file, and check that every sheet has the same number of
real and synthetic comments (real breakdown total vs. DG/SAG/SAGII counts).

Each sheet follows a fixed layout:
    Opening Post
    Title: ...
    Body: ...
    Real breakdown: "<n> clinician, <m> layperson (<n+m> total)"
    (blank)
    Real Comments
    Author Type | Comment
    ...
    Synthetic: DG (Discourse Generation)
    Author Type | Comment
    ...
    Synthetic: SAG (Single Advice Generation)
    Author Type | Comment
    ...
    Synthetic: SAGII (SAG + Incremental Info)
    Author Type | Comment
    ...

Real comments are extracted verbatim, one JSON entry per non-empty cell in the
"Real Comments" section. In 21/100 sheets the number of real-comment cells is
lower than the "Real breakdown" total (i.e. several real comments were merged
into fewer cells upstream, in the source spreadsheet). There's no reliable way
to auto-detect where those boundaries were meant to be (splitting on blank
lines only reconciles the count for 1 of the 21 mismatched sheets - the rest
would need fabricated boundaries), so mismatches are only flagged, never
silently patched. See qc_comment_counts.csv for the full list.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import openpyxl

SYNTHETIC_METHODS = {
    "DG": "Synthetic: DG",
    "SAG": "Synthetic: SAG (",
    "SAGII": "Synthetic: SAGII",
}
BREAKDOWN_RE = re.compile(r"(\d+)\s*clinician,\s*(\d+)\s*layperson\s*\((\d+)\s*total\)")


def classify_section(label):
    if label == "Real Comments":
        return "real"
    if not isinstance(label, str):
        return None
    for method, prefix in SYNTHETIC_METHODS.items():
        if label.startswith(prefix):
            return method
    return None


def parse_sheet(ws, post_id):
    title = None
    body = None
    breakdown = {"clinician": None, "layperson": None, "total": None}

    real_comments = []
    synthetic_comments = {"DG": [], "SAG": [], "SAGII": []}

    section = None
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, values_only=True):
        label, value = row[0], row[1] if len(row) > 1 else None

        if label == "Title:":
            title = value
            continue
        if label == "Body:":
            body = value
            continue
        if label == "Real breakdown:":
            m = BREAKDOWN_RE.search(value or "")
            if m:
                breakdown = {
                    "clinician": int(m.group(1)),
                    "layperson": int(m.group(2)),
                    "total": int(m.group(3)),
                }
            continue
        if label == "Author Type":
            continue

        new_section = classify_section(label)
        if new_section is not None:
            section = new_section
            continue

        if section is not None and value is not None:
            entry = {"author_type": label, "comment": value}
            if section == "real":
                real_comments.append(entry)
            else:
                synthetic_comments[section].append(entry)

    return {
        "post_id": post_id,
        "sheet_name": ws.title,
        "title": title,
        "body": body,
        "real_breakdown": breakdown,
        "real_comments": real_comments,
        "synthetic_comments": synthetic_comments,
    }


def check_counts(post):
    expected = post["real_breakdown"]["total"]
    counts = {
        "real": len(post["real_comments"]),
        "DG": len(post["synthetic_comments"]["DG"]),
        "SAG": len(post["synthetic_comments"]["SAG"]),
        "SAGII": len(post["synthetic_comments"]["SAGII"]),
    }
    all_match = expected is not None and all(c == expected for c in counts.values())
    return all_match, expected, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=str, default="../../input/round3_clinician_synthetic_comments.xlsx")
    parser.add_argument("--output_json", type=str, default="../../output/3_inference/posts.json")
    parser.add_argument("--output_qc_csv", type=str, default="../../output/3_inference/qc_comment_counts.csv")
    args = parser.parse_args()

    wb = openpyxl.load_workbook(args.input_path, data_only=True)

    posts = []
    qc_rows = []
    n_mismatch = 0
    for post_id, sheet_name in enumerate(wb.sheetnames, start=1):
        ws = wb[sheet_name]
        post = parse_sheet(ws, post_id)
        posts.append(post)

        all_match, expected, counts = check_counts(post)
        if not all_match:
            n_mismatch += 1
        qc_rows.append({
            "post_id": post_id,
            "sheet_name": sheet_name,
            "expected_total": expected,
            "real_count": counts["real"],
            "DG_count": counts["DG"],
            "SAG_count": counts["SAG"],
            "SAGII_count": counts["SAGII"],
            "match": all_match,
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

    print(f"Parsed {len(posts)} sheets -> {out_json}")
    print(f"QC report -> {out_csv}")
    print(f"{len(posts) - n_mismatch}/{len(posts)} sheets have matching real/synthetic comment counts.")
    if n_mismatch:
        print(f"\n{n_mismatch} sheets have FEWER real comments than the stated breakdown total")
        print("(real comments merged into fewer cells upstream; not auto-split — see qc_comment_counts.csv):")
        for row in qc_rows:
            if not row["match"]:
                print(f"  post {row['post_id']:>3} | {row['sheet_name']!r:35} "
                      f"expected={row['expected_total']} real={row['real_count']} "
                      f"DG={row['DG_count']} SAG={row['SAG_count']} SAGII={row['SAGII_count']}")


if __name__ == "__main__":
    main()
