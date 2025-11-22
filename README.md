# Candidate Evaluation Classifier

A cross-encoder model for evaluating candidate-qualification matches. Given a qualification requirement (Q) and a candidate description (C), the model classifies the match into one of three categories.

## Labels

| Label | Description |
|-------|-------------|
| `NOT_MATCH` | Candidate does not meet the qualification |
| `PARTIAL_MATCH` | Candidate partially meets the qualification |
| `MATCH` | Candidate fully meets the qualification |

## Installation

Requires Python 3.13+. Install dependencies using `uv`:

```bash
uv sync
```

## Usage

### Training

**Option 1: Synthetic data (quick PoC)**

Train on 1,000 synthetic Q-C-label examples:

```bash
uv run python candidate_eval_poc.py
```

This will train for 2 epochs and save to `./candidate_eval_cross_encoder_poc`.

**Option 2: HuggingFace dataset (real resumes)**

Train on the [netsol/resume-score-details](https://huggingface.co/datasets/netsol/resume-score-details) dataset (~1,000 real resume-job pairs):

```bash
uv run python candidate_eval_hf_dataset.py
```

This will:
- Download real resume/job description pairs from HuggingFace
- Map match scores to labels (NOT_MATCH, PARTIAL_MATCH, MATCH)
- Train for 3 epochs and save to `./candidate_eval_hf_model`

### Inference

Run inference on the trained model:

```bash
# Use local fine-tuned checkpoint
uv run python inference.py --model ./candidate_eval_cross_encoder_poc

# Interactive mode - enter your own examples
uv run python inference.py --model ./candidate_eval_cross_encoder_poc --interactive

# Use base model from HuggingFace (no fine-tuning)
uv run python inference.py --model distilbert-base-uncased
```

## Project Structure

```
candidate_eval_poc.py         # Training script with synthetic data
candidate_eval_hf_dataset.py  # Training script using HuggingFace dataset
inference.py                  # Standalone inference script
candidate_eval_cross_encoder_poc/  # Model from synthetic training
candidate_eval_hf_model/      # Model from HuggingFace dataset training
Notes.md                      # Technical notes and Q&A
pyproject.toml                # Project dependencies
```

## How It Works

The model uses a cross-encoder architecture:

1. Input is formatted as `[CLS] qualification [SEP] candidate [SEP]`
2. DistilBERT encodes the combined input
3. The `[CLS]` token embedding is projected to 3 logits via a classification head
4. Softmax converts logits to probabilities

See [Notes.md](Notes.md) for more technical details.
