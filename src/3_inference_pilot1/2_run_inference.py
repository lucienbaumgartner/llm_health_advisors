"""
Run the trained empathy classifiers (Emotional Reactions / Interpretations /
Explorations, from ../2_training) on every real and synthetic comment in
posts.json. Posts themselves are not classified - they're only used as the
seeker_post context each comment is paired with.

This reimplements batched inference directly (rather than reusing
../2_training/empathy_classifier.py's EmpathyClassifier) because that class's
predict_empathy() loops over DataLoader batches but only returns the *last*
batch's predictions - harmless there since test.py only ever calls it with
single-example batches, but wrong for batched inference over many comments.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaTokenizer

TRAINING_DIR = Path(__file__).resolve().parent.parent / "2_training"
sys.path.insert(0, str(TRAINING_DIR))

from models.models import BiEncoderAttentionWithRationaleClassification  # noqa: E402

MECHANISMS = ["ER", "IP", "EX"]
MAX_LEN = 64  # must match training (src/2_training/train.py --max_len default)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_records(posts):
    records = []
    for post in posts:
        seeker_post = post["body"] or ""
        for c in post["real_comments"]:
            records.append({
                "post_id": post["post_id"],
                "sheet_name": post["sheet_name"],
                "source": "real",
                "method": None,
                "author_type": c["author_type"],
                "seeker_post": seeker_post,
                "response_post": c["comment"],
            })
        for method, comments in post["synthetic_comments"].items():
            for c in comments:
                records.append({
                    "post_id": post["post_id"],
                    "sheet_name": post["sheet_name"],
                    "source": "synthetic",
                    "method": method,
                    "author_type": c["author_type"],
                    "seeker_post": seeker_post,
                    "response_post": c["comment"],
                })
    return records


def load_model(checkpoint_path, device):
    model = BiEncoderAttentionWithRationaleClassification()
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(records, models, tokenizer, device, batch_size):
    seeker_posts = [r["seeker_post"] for r in records]
    response_posts = [r["response_post"] for r in records]

    enc_sp = tokenizer(seeker_posts, add_special_tokens=True, max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_attention_mask=True)
    enc_rp = tokenizer(response_posts, add_special_tokens=True, max_length=MAX_LEN,
                        padding="max_length", truncation=True, return_attention_mask=True)

    dataset = TensorDataset(
        torch.tensor(enc_sp["input_ids"]), torch.tensor(enc_sp["attention_mask"]),
        torch.tensor(enc_rp["input_ids"]), torch.tensor(enc_rp["attention_mask"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = {m: [] for m in MECHANISMS}
    probs = {m: [] for m in MECHANISMS}

    with torch.no_grad():
        for batch in loader:
            input_ids_sp, mask_sp, input_ids_rp, mask_rp = (t.to(device) for t in batch)
            for mech in MECHANISMS:
                logits_empathy, _logits_rationale = models[mech](
                    input_ids_SP=input_ids_sp, input_ids_RP=input_ids_rp,
                    attention_mask_SP=mask_sp, attention_mask_RP=mask_rp,
                )
                batch_probs = F.softmax(logits_empathy, dim=-1).cpu().numpy()
                batch_preds = batch_probs.argmax(axis=1)
                preds[mech].extend(batch_preds.tolist())
                probs[mech].extend(batch_probs.tolist())

    for mech in MECHANISMS:
        for i, r in enumerate(records):
            r[f"{mech}_label"] = preds[mech][i]
            r[f"{mech}_prob_0"] = probs[mech][i][0]
            r[f"{mech}_prob_1"] = probs[mech][i][1]
            r[f"{mech}_prob_2"] = probs[mech][i][2]

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_json", type=str, default="../../output/3_inference/posts.json")
    parser.add_argument("--output_csv", type=str, default="../../output/3_inference/inference_results.csv")
    parser.add_argument("--ER_model_path", type=str, default="../../output/2_training/reddit_ER.pth")
    parser.add_argument("--IP_model_path", type=str, default="../../output/2_training/reddit_IP.pth")
    parser.add_argument("--EX_model_path", type=str, default="../../output/2_training/reddit_EX.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = get_device()
    print("device =", device)

    posts = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    records = build_records(posts)
    print(f"{len(records)} comments to classify "
          f"({sum(r['source'] == 'real' for r in records)} real, "
          f"{sum(r['source'] == 'synthetic' for r in records)} synthetic)")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    models = {
        "ER": load_model(args.ER_model_path, device),
        "IP": load_model(args.IP_model_path, device),
        "EX": load_model(args.EX_model_path, device),
    }

    records = run_inference(records, models, tokenizer, device, args.batch_size)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"Wrote predictions -> {out_path}")


if __name__ == "__main__":
    main()
