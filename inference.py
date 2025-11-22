# inference.py
#
# Standalone inference script for candidate evaluation classifier.
# Supports both HuggingFace hub models and local checkpoints.
#
# Usage:
#   # Use base model (no fine-tuning) from HuggingFace
#   uv run python inference.py --model distilbert-base-uncased
#
#   # Use local fine-tuned checkpoint
#   uv run python inference.py --model ./candidate_eval_cross_encoder_poc
#
#   # Interactive mode - enter your own examples
#   uv run python inference.py --model ./candidate_eval_cross_encoder_poc --interactive

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


# Label schema (must match training)
LABEL2ID: Dict[str, int] = {
    "NOT_MATCH": 0,
    "PARTIAL_MATCH": 1,
    "MATCH": 2,
    "FORBIDDEN": 3,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 256


def find_checkpoint_path(model_path: str) -> str:
    """
    Find the actual model checkpoint path.
    Handles cases where checkpoints are in subdirectories (e.g., checkpoint-200/).
    """
    path = Path(model_path)

    # If it's not a local directory, assume it's a HuggingFace model name
    if not path.exists():
        return model_path

    # Check if config.json exists directly
    if (path / "config.json").exists():
        return model_path

    # Look for checkpoint subdirectories
    checkpoint_dirs = sorted(path.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[1]))
    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"Found checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)

    return model_path


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer from HuggingFace hub or local checkpoint.

    For base models (not fine-tuned), we need to specify num_labels.
    For local checkpoints, the config is already saved.
    """
    print(f"Loading model from: {model_path}")

    # Find actual checkpoint path (handles checkpoint-* subdirs)
    actual_model_path = find_checkpoint_path(model_path)
    is_local = Path(actual_model_path).exists()

    # For local checkpoints, tokenizer may not be saved - load from base model
    if is_local:
        try:
            # Try loading tokenizer from checkpoint first
            tokenizer = AutoTokenizer.from_pretrained(actual_model_path, use_fast=True)
        except Exception:
            # Tokenizer not saved in checkpoint - get base model name from config
            print("Tokenizer not found in checkpoint, loading from base model...")
            config = AutoConfig.from_pretrained(actual_model_path)
            # _name_or_path often points to checkpoint path, not original model
            # Use model_type to determine the correct base tokenizer
            model_type = getattr(config, "model_type", "distilbert")
            if model_type == "distilbert":
                base_model = "distilbert-base-uncased"
            else:
                base_model = getattr(config, "_name_or_path", "distilbert-base-uncased")
            print(f"Using tokenizer from: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(actual_model_path, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    # Try loading as a fine-tuned model first (has correct num_labels)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(actual_model_path)
        # Check if it has the right number of labels
        if model.config.num_labels != len(LABEL2ID):
            print(f"Warning: Model has {model.config.num_labels} labels, expected {len(LABEL2ID)}")
            print("Loading as base model with correct label config...")
            raise ValueError("Label mismatch")
    except Exception as e:
        if "num_labels" not in str(e) and "Label mismatch" not in str(e):
            # Real error, not just label mismatch
            raise
        # Load as base model with our label config
        print("Loading as base model (not fine-tuned for this task)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            actual_model_path,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Move to best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(f"Model loaded on device: {device}")

    return model, tokenizer


def predict_single(qualification: str, candidate: str, tokenizer, model) -> Dict:
    """
    Run inference on one (Q, C) pair.
    """
    text = f"Qualification: {qualification}\nCandidate: {candidate}"
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    pred_id = int(np.argmax(probs))
    pred_label = ID2LABEL[pred_id]
    prob_dict = {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

    return {
        "label": pred_label,
        "probs": prob_dict,
    }


def print_prediction(qualification: str, candidate: str, result: Dict):
    """Pretty print a prediction result."""
    print("-" * 60)
    print(f"Qualification: {qualification}")
    print(f"Candidate: {candidate}")
    print(f"\nPredicted: {result['label']}")
    print("Probabilities:")
    for label, prob in sorted(result['probs'].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"  {label:15} {prob:.4f} {bar}")


def run_demo_examples(model, tokenizer):
    """Run inference on a set of demo examples."""
    examples = [
        # FORBIDDEN - discriminatory qualification
        {
            "qualification": "Only male applicants will be considered for this senior engineering role.",
            "candidate": "Senior engineer with 10 years of experience in backend systems, based in San Jose, CA.",
            "expected": "FORBIDDEN",
        },
        # FORBIDDEN - age discrimination
        {
            "qualification": "Looking for a young candidate under 30 years old.",
            "candidate": "Software developer with 5 years of Python experience.",
            "expected": "FORBIDDEN",
        },
        # MATCH - location matches
        {
            "qualification": "Must be located in commutable distance to South Bay, California.",
            "candidate": "Lives in San Jose, California. Senior software engineer with 5 years of experience.",
            "expected": "MATCH",
        },
        # MATCH - skill matches
        {
            "qualification": "Must have strong experience in Python.",
            "candidate": "Worked 5 years as a backend engineer using Python, with multiple projects in data processing and APIs.",
            "expected": "MATCH",
        },
        # PARTIAL_MATCH - willing to relocate
        {
            "qualification": "Must be located in commutable distance to South Bay, California.",
            "candidate": "Lives in New York, but open to relocation to California in 6-12 months.",
            "expected": "PARTIAL_MATCH",
        },
        # NOT_MATCH - wrong location, no relocation
        {
            "qualification": "Must be located in commutable distance to South Bay, California.",
            "candidate": "Lives in Boston, no plans to relocate. Works as a product manager.",
            "expected": "NOT_MATCH",
        },
        # NOT_MATCH - wrong skill
        {
            "qualification": "Must have strong experience in Python.",
            "candidate": "Strong experience in Java only, no mention of Python.",
            "expected": "NOT_MATCH",
        },
    ]

    print("\n" + "=" * 60)
    print("DEMO EXAMPLES")
    print("=" * 60)

    correct = 0
    for ex in examples:
        result = predict_single(ex["qualification"], ex["candidate"], tokenizer, model)
        print_prediction(ex["qualification"], ex["candidate"], result)

        match_str = "✓" if result["label"] == ex["expected"] else "✗"
        print(f"Expected: {ex['expected']} {match_str}")
        if result["label"] == ex["expected"]:
            correct += 1

    print("\n" + "=" * 60)
    print(f"Accuracy on demo examples: {correct}/{len(examples)} ({100*correct/len(examples):.1f}%)")
    print("=" * 60)


def run_interactive(model, tokenizer):
    """Interactive mode - enter your own examples."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter qualification and candidate texts to get predictions.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        print("\n")
        qualification = input("Qualification (or 'quit'): ").strip()
        if qualification.lower() in ("quit", "exit", "q"):
            break

        candidate = input("Candidate: ").strip()
        if not candidate:
            print("Candidate cannot be empty.")
            continue

        result = predict_single(qualification, candidate, tokenizer, model)
        print_prediction(qualification, candidate, result)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with candidate evaluation classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./candidate_eval_cross_encoder_poc",
        help="Model path: HuggingFace model name or local checkpoint path (default: ./candidate_eval_cross_encoder_poc)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to enter custom examples",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Run demo examples
    run_demo_examples(model, tokenizer)

    # Interactive mode if requested
    if args.interactive:
        run_interactive(model, tokenizer)


if __name__ == "__main__":
    main()
