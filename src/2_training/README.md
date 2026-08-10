# 2_training — Empathy Classifier

Code and public data vendored from [behavioral-data/Empathy-Mental-Health](https://github.com/behavioral-data/Empathy-Mental-Health)
(commit as of 2026-08-10), which implements the multi-task RoBERTa bi-encoder
model from:

> Sharma, A., Miner, A. S., Atkins, D. C., & Althoff, T. (2020). A Computational
> Approach to Understanding Empathy Expressed in Text-Based Mental Health
> Support. *EMNLP*.

Usage terms are in [licenses/](licenses/) (non-commercial academic use,
citation required — see `Althoff academic license.docx`).

## What's here vs. what was left out

Only the **Reddit** portion of the original repo is included:

- `data/emotional-reactions-reddit.csv`
- `data/interpretations-reddit.csv`
- `data/explorations-reddit.csv`
- `data/sample_input_ER.csv`, `data/sample_test_input.csv` (small samples for
  smoke-testing the pipeline)

The **TalkLife** portion of the original dataset is not distributed in the
source repo at all (it requires contacting research@talklife.co directly) and
is deliberately **not used** for this project. Nothing here depends on it.

The corpus built in [`../1_preprocessing`](../1_preprocessing) is not used as
training input here — it's reserved for later machine annotation with the
trained classifier, not for training it.

## Setup

The original repo pinned Python 3.7 / torch 1.4.0 / transformers 2.7.0, which
have no arm64 (Apple Silicon) wheels. The code here has been **patched to run
on a modern stack** (torch 2.13 / transformers 5.15) so it works locally with
GPU acceleration via Apple's MPS backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`train.py`/`test.py` pick `cuda` > `mps` > `cpu` automatically, so this also
runs unmodified on a CUDA machine.

Patches applied to the vendored code (relative to the original repo):
- `pad_to_max_length=True` / `return_attention_masks=True` → `padding='max_length', truncation=True, return_attention_mask=True`
  (renamed/removed tokenizer kwargs).
- `tokenizer.encode_plus(...)` / `tokenizer.batch_encode_plus(...)` → `tokenizer(...)`
  (those methods were removed from the public API; calling the tokenizer directly is the modern equivalent).
- `from transformers import AdamW` → `from torch.optim import AdamW` (removed from `transformers`).
- Dropped unused `GPT2Model` / `AutoModelWithLMHead` / `AutoTokenizer` imports in `models/models.py`
  (unused, and `AutoModelWithLMHead` no longer exists).
- Added an `mps` branch to the `cuda`/`cpu` device selection in `train.py` and `test.py`.

Verified end-to-end with a 1-epoch smoke run on `data/sample_input_ER.csv` on
MPS (loss decreased, checkpoint saved). `models/models.py` still pulls
`roberta-base` weights from the legacy `s3.amazonaws.com/models.huggingface.co`
mirror baked into the vendored `models/file_utils.py` — confirmed reachable,
but it's a ~500MB one-time download the first time you instantiate the model.

## Preparing data

Convert a raw annotated CSV into the tokenized format the model expects:

```bash
python3 process_data.py --input_path data/emotional-reactions-reddit.csv --output_path data/emotional-reactions-reddit.model.csv
```

## Training

Create an output directory, then train one model per empathy mechanism
(Emotional Reactions / Interpretations / Explorations):

```bash
mkdir -p ../../output/2_training

python3 train.py \
  --train_path=data/emotional-reactions-reddit.model.csv \
  --lr=2e-5 --batch_size=32 --lambda_EI=1.0 --lambda_RE=0.5 \
  --save_model --save_model_path=../../output/2_training/reddit_ER.pth

python3 train.py \
  --train_path=data/interpretations-reddit.model.csv \
  --lr=2e-5 --batch_size=32 --lambda_EI=1.0 --lambda_RE=0.5 \
  --save_model --save_model_path=../../output/2_training/reddit_IP.pth

python3 train.py \
  --train_path=data/explorations-reddit.model.csv \
  --lr=2e-5 --batch_size=32 --lambda_EI=1.0 --lambda_RE=0.5 \
  --save_model --save_model_path=../../output/2_training/reddit_EX.pth
```

(Each `data/*-reddit.csv` needs to go through `process_data.py` first, same as
the sample above.)

## Testing / inference

```bash
python3 test.py \
  --input_path data/sample_test_input.csv \
  --output_path data/sample_test_output.csv \
  --ER_model_path ../../output/2_training/reddit_ER.pth \
  --IP_model_path ../../output/2_training/reddit_IP.pth \
  --EX_model_path ../../output/2_training/reddit_EX.pth
```
