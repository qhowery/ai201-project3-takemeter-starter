#!/usr/bin/env python3
"""Milestone 5 — Fine-tune DistilBERT and evaluate on locked test split."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from torch import nn

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "r_nba_labeled.csv"
BASELINE_PATH = ROOT / "baseline_evaluation.json"
OUT_JSON = ROOT / "evaluation_results.json"
OUT_CM = ROOT / "confusion_matrix.png"
MODEL_DIR = ROOT / "takemeter-model"

LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "news_updates": 2,
    "meta_community": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
MODEL_NAME = "distilbert-base-uncased"


def load_splits():
    df = pd.read_csv(CSV_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def make_datasets(train_df, val_df, test_df, tokenizer):
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    def to_hf(df_split):
        ds = Dataset.from_pandas(
            df_split[["text", "label_id"]].rename(columns={"label_id": "labels"})
        )
        return ds.map(tokenize, batched=True)

    return to_hf(train_df), to_hf(val_df), to_hf(test_df)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    counts = train_df["label_id"].value_counts().sort_index()
    # Inverse frequency — helps minority labels like hot_take
    weights = len(train_df) / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def main() -> None:
    print("Loading data...")
    train_df, val_df, test_df = load_splits()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device_note = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    print(f"Device: {device_note}")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    train_dataset, val_dataset, test_dataset = make_datasets(
        train_df, val_df, test_df, tokenizer
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class_weights = build_class_weights(train_df)
    print("Class weights:", dict(zip(LABEL_MAP.keys(), class_weights.tolist())))

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=30,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        use_cpu=not (use_cuda or use_mps),
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting fine-tuning (5 epochs, class-weighted)...")
    trainer.train()
    print("Fine-tuning complete. Running test evaluation...")

    ft_output = trainer.predict(test_dataset)
    ft_pred_ids = np.argmax(ft_output.predictions, axis=-1)
    ft_true_ids = ft_output.label_ids
    ft_accuracy = accuracy_score(ft_true_ids, ft_pred_ids)

    label_names = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]
    report_text = classification_report(
        ft_true_ids, ft_pred_ids, target_names=label_names, zero_division=0
    )
    report_dict = classification_report(
        ft_true_ids,
        ft_pred_ids,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    print(f"\nFine-tuned accuracy: {ft_accuracy:.3f}")
    print("\nPer-class metrics (fine-tuned):")
    print(report_text)

    cm = confusion_matrix(ft_true_ids, ft_pred_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Fine-Tuned Model — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(OUT_CM, dpi=150)
    plt.close()
    print(f"Saved {OUT_CM}")

    baseline_accuracy = None
    if BASELINE_PATH.exists():
        baseline_accuracy = json.loads(BASELINE_PATH.read_text())["baseline_accuracy"]

    pred_counts = pd.Series(ft_pred_ids).value_counts(normalize=True)
    max_pred_share = float(pred_counts.max())

    results = {
        "baseline_accuracy": baseline_accuracy,
        "finetuned_accuracy": round(float(ft_accuracy), 4),
        "improvement": round(float(ft_accuracy - baseline_accuracy), 4)
        if baseline_accuracy is not None
        else None,
        "test_set_size": len(test_df),
        "label_map": LABEL_MAP,
        "model": MODEL_NAME,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
        "max_prediction_class_share": round(max_pred_share, 4),
        "split": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "random_state": 42,
        },
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT_JSON}")

    if baseline_accuracy is not None:
        delta = ft_accuracy - baseline_accuracy
        print("\n" + "=" * 50)
        print("RESULTS COMPARISON")
        print("=" * 50)
        print(f"{'Zero-shot baseline (Groq)':<35} {baseline_accuracy:>8.3f}")
        print(f"{'Fine-tuned DistilBERT':<35} {ft_accuracy:>8.3f}")
        print(f"{'Improvement':<35} {delta:>+8.3f}")
        target = baseline_accuracy + 0.08
        print(f"\nTarget (baseline + 0.08): {target:.3f} — {'MET' if ft_accuracy >= target else 'NOT MET'}")


if __name__ == "__main__":
    main()
