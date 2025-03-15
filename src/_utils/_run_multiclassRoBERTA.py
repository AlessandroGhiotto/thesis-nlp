import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
import json
from datetime import datetime
import time
from evaluate import evaluator
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "roberta-base"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def combine_datasets(real_df=None, synth_df=None, synth_ratio=0.0, max_samples=500):
    """Combine real and synthetic datasets to create a training dataset."""
    combined_df = pd.DataFrame()

    # combine real and synthetic datasets
    if real_df is not None and synth_df is not None and synth_ratio > 0:
        num_synth = int(max_samples * synth_ratio)
        num_real = max_samples - num_synth
        real_sample = real_df.sample(n=min(num_real, len(real_df)), random_state=42)
        synth_sample = synth_df.sample(n=min(num_synth, len(synth_df)), random_state=42)
        combined_df = pd.concat([real_sample, synth_sample], ignore_index=True)
    # only real data
    elif real_df is not None:
        combined_df = real_df.sample(n=min(max_samples, len(real_df)), random_state=42)
    # only synthetic data
    elif synth_df is not None:
        combined_df = synth_df.sample(
            n=min(max_samples, len(synth_df)), random_state=42
        )
    else:
        raise ValueError("At least one dataset must be provided.")

    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess_data(df, label2id):
    def tokenize_function(examples):
        # Tokenize the text and add labels
        encoding = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128
        )
        encoding["labels"] = [label2id[label] for label in examples["label"]]
        return encoding

    return df.map(tokenize_function, batched=True, remove_columns=df.column_names)


def compute_metrics(p):
    # Single-label multi-class classification
    y_pred = np.argmax(p.predictions, axis=1)
    y_true = p.label_ids

    return {
        "f1": f1_score(y_true, y_pred, average="micro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def train_model(train_df, dev_df, labels, output_dir):
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        problem_type="single_label_classification",  # single-label multi-class classification
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_df,
        eval_dataset=dev_df,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"Training completed for experiment saved in: {output_dir}")


def evaluation(test_df, label2id, modelname):
    task_evaluator = evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=modelname,
        data=data["train"],
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        label_mapping=label2id,
    )
    return eval_results


def save_log(
    output_dir, experiment_name, train_size, dev_size, synth_ratio, train_time, metrics
):
    log = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "train_size": train_size,
        "dev_size": dev_size,
        "synthetic_ratio": synth_ratio,
        "train_time_seconds": train_time,
        "metrics": metrics,
    }
    log_path = os.path.join(output_dir, "log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=4)


def prepare_and_train(
    real_df=None,
    synth_df=None,
    dev_df=None,
    synth_ratio=0.0,
    max_samples=500,
    output_dir="experiment_output",
):
    if dev_df is None:
        raise ValueError("A development dataset (dev_df) must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    # combine datasets
    combined_df = combine_datasets(real_df, synth_df, synth_ratio, max_samples)

    # save datasets for reproducibility
    train_path = os.path.join(output_dir, "train.csv")
    dev_path = os.path.join(output_dir, "dev.csv")
    combined_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)

    # preprocess data
    labels = combined_df["label"].unique().tolist()
    label2id = {label: idx for idx, label in enumerate(labels)}
    train_dataset = preprocess_data(Dataset.from_pandas(combined_df), label2id)
    dev_dataset = preprocess_data(Dataset.from_pandas(dev_df), label2id)

    # train
    start_time = time.time()
    labels_ids = [label2id[label] for label in labels]
    train_model(train_dataset, dev_dataset, labels_ids, output_dir)
    end_time = time.time()

    # Metrics can be logged here if available post-training
    metrics = {
        "f1": 0.85,
        "f1_macro": 0.80,
        "accuracy": 0.82,
    }  # Placeholder, replace with actual metrics

    save_log(
        output_dir,
        os.path.basename(output_dir),
        len(combined_df),
        len(dev_df),
        synth_ratio,
        end_time - start_time,
        metrics,
    )

    print(f"Experiment saved in: {output_dir}")


# Example usage:
# prepare_and_train(real_df=real_df, synth_df=random_df, dev_df=dev_df, synth_ratio=0.3, max_samples=500, output_dir="random_aug_experiment")
# prepare_and_train(real_df=None, synth_df=targeted_df, dev_df=dev_df, max_samples=500, output_dir="targeted_synth_only")
# prepare_and_train(real_df=real_df, dev_df=dev_df, max_samples=500, output_dir="real_only_experiment")
