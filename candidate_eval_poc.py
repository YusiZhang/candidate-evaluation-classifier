# candidate_eval_poc.py
#
# Proof-of-concept: cross-encoder for candidate evaluation (Q, C -> 3 labels)
# - Generates 1,000 synthetic Q-C-label examples
# - Trains a DistilBERT-based classifier on CPU
# - Prints basic metrics and a sample prediction
#
# Run:
#   python candidate_eval_poc.py

import random
from typing import Dict, List

import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# -------------------------------------------------------------------
# 1. Label schema
# -------------------------------------------------------------------
LABEL2ID: Dict[str, int] = {
    "NOT_MATCH": 0,
    "PARTIAL_MATCH": 1,
    "MATCH": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


# -------------------------------------------------------------------
# 2. Synthetic data generation (1,000 Q-C pairs)
# -------------------------------------------------------------------
def generate_synthetic_example() -> Dict[str, str]:
    """
    Generate a synthetic (qualification, candidate, label) triple.

    Labels:
    - MATCH: candidate fully meets the qualification
    - PARTIAL_MATCH: candidate partially meets the qualification
    - NOT_MATCH: candidate does not meet the qualification
    """
    # Randomly choose label
    patterns = ["MATCH", "PARTIAL_MATCH", "NOT_MATCH"]
    label = random.choice(patterns)

    # Generate normal qualifications and vary candidate fit.
    locations = ["South Bay, California", "New York City", "Austin, Texas"]
    cities_near_south_bay = ["San Jose", "Sunnyvale", "Mountain View"]
    cities_far = ["New York", "Boston", "Chicago"]

    skills = ["Python", "Java", "JavaScript", "SQL"]
    target_skill = "Python"

    # Normal (non-biased) qualification types
    q_type = random.choice(["location", "skill", "experience"])

    if q_type == "location":
        qualification = (
            f"Must be located in commutable distance to {random.choice(locations)}."
        )
        if label == "MATCH":
            city = random.choice(cities_near_south_bay)
            candidate = f"Lives in {city}, California. Senior software engineer with 5 years of experience."
        elif label == "PARTIAL_MATCH":
            city = random.choice(cities_far)
            candidate = f"Lives in {city}, but open to relocation to California in 6-12 months."
        else:  # NOT_MATCH
            city = random.choice(cities_far)
            candidate = f"Lives in {city}, no plans to relocate. Works as a product manager."

    elif q_type == "skill":
        qualification = f"Must have strong experience in {target_skill}."
        if label == "MATCH":
            candidate = (
                f"Worked 5 years as a backend engineer using {target_skill}, "
                f"with multiple projects in data processing and APIs."
            )
        elif label == "PARTIAL_MATCH":
            candidate = (
                f"Has 1 year of professional {target_skill} experience and 3 years with Java and SQL."
            )
        else:  # NOT_MATCH
            other_skill = random.choice([s for s in skills if s != target_skill])
            candidate = (
                f"Strong experience in {other_skill} only, no mention of {target_skill}."
            )

    else:  # experience
        qualification = "Must have at least 3 years of software engineering experience."
        if label == "MATCH":
            candidate = (
                "Software engineer with 5 years of experience in backend and distributed systems."
            )
        elif label == "PARTIAL_MATCH":
            candidate = (
                "Junior engineer with 1.5 years of experience in frontend development."
            )
        else:  # NOT_MATCH
            candidate = "Recent graduate with internship experience only, no full-time roles."

    # columns: qualification, candidate, label are literal strings, they will be tokenized later
    return {
        "qualification": qualification,
        "candidate": candidate,
        "label": label,
    }


def build_synthetic_dataset(n_examples: int = 1000) -> DatasetDict:
    """
    Build a DatasetDict with 'train' and 'validation' splits
    from synthetic data.
    """
    examples: List[Dict[str, str]] = [generate_synthetic_example() for _ in range(n_examples)]

    dataset = Dataset.from_list(examples)
    # 80% train, 20% validation
    ds_split = dataset.train_test_split(test_size=0.2, seed=42)

    return DatasetDict(
        {
            "train": ds_split["train"],
            "validation": ds_split["test"],
        }
    )


# -------------------------------------------------------------------
# 3. Tokenization & preprocessing
# -------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256  # smaller for laptop friendliness


def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def preprocess_function(examples, tokenizer):
    texts = []
    for q, c in zip(examples["qualification"], examples["candidate"]):
        # Cross-encoder input: both Q and C in a single sequence
        text = f"Qualification: {q}\nCandidate: {c}"
        texts.append(text)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    labels = [LABEL2ID[label] for label in examples["label"]]
    encodings["labels"] = labels

    return encodings


# -------------------------------------------------------------------
# 4. Metrics
# -------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# -------------------------------------------------------------------
# 5. Main training & inference
# -------------------------------------------------------------------
def main():
    # 1) Build synthetic dataset (1,000 Q-C pairs)
    dataset = build_synthetic_dataset(n_examples=1000)
    print(dataset)

    # 2) Tokenizer
    tokenizer = build_tokenizer()

    # 3) Preprocess
    tokenized = dataset.map(
        lambda batch: preprocess_function(batch, tokenizer),
        batched=True,
        remove_columns=["qualification", "candidate", "label"],
    )
    """
    Before map:
    qualification	    candidate	             label
    "Python 5 years"	"I have 6 years Python"	 1
    After map:
    input_ids	        attention_mask	labels
    [101, 2034, ...]	[1, 1, 1, ...]	1
    """

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 5) Training args (CPU-friendly)
    training_args = TrainingArguments(
        # Directory to save model checkpoints and outputs
        output_dir="./candidate_eval_cross_encoder_poc",
        # Number of samples per training step per device (lower = less memory, slower)
        per_device_train_batch_size=4,
        # Number of samples per evaluation step (can be larger since no gradients stored)
        per_device_eval_batch_size=8,
        # Step size for optimizer; 2e-5 is typical for fine-tuning transformers
        learning_rate=2e-5,
        # Number of complete passes through the training data (small for PoC)
        num_train_epochs=2,
        # L2 regularization factor to prevent overfitting
        weight_decay=0.01,
        # Log training metrics (loss, etc.) every N steps
        logging_steps=20,
        # Run evaluation on validation set after each epoch
        eval_strategy="epoch",
        # Save model checkpoint after each epoch
        save_strategy="epoch",
        # After training, load the checkpoint with best validation metric
        load_best_model_at_end=True,
        # Use macro F1 score to determine "best" model
        metric_for_best_model="macro_f1",
        # Higher macro_f1 = better (vs metrics like loss where lower is better)
        greater_is_better=True,
        # Disable 16-bit floating point training (requires GPU with tensor cores)
        fp16=False,
        # Keep only 1 checkpoint to save disk space (deletes older ones)
        save_total_limit=1,
        # Disable logging to external services (wandb, tensorboard, etc.)
        report_to="none",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # 7) Train
    trainer.train()

    # 7.5) Save tokenizer to output directory so inference can load it
    tokenizer.save_pretrained(training_args.output_dir)

    # 8) Evaluate
    metrics = trainer.evaluate()
    print("\n=== Evaluation metrics on validation set ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"{k}: {v}")

    # 9) Quick inference demo on a custom example
    print("\n=== Inference demo ===")
    q_demo = "Only male applicants will be considered for this senior engineering role."
    c_demo = "Senior engineer with 10 years of experience in backend systems, based in San Jose, CA."
    prediction = predict_single(q_demo, c_demo, tokenizer, model)
    print("Qualification:", q_demo)
    print("Candidate:", c_demo)
    print("Predicted:", prediction)


def predict_single(qualification: str, candidate: str, tokenizer, model):
    """
    Run inference on one (Q, C) pair using an already-loaded tokenizer & model.
    """
    import torch

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


if __name__ == "__main__":
    main()
