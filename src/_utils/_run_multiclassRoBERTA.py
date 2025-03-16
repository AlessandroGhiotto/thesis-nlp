import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
import json
from datetime import datetime
import time
import shutil
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from src._utils._helpers import set_seed

MODEL_NAME = "roberta-large"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, verbose=False)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def combine_datasets(real_df=None, synth_df=None, synth_ratio=0.0, max_samples=500):
    """Combine real and synthetic datasets to create a training dataset."""
    combined_df = pd.DataFrame()

    # combine real and synthetic datasets
    if real_df is not None and synth_df is not None and 0 < synth_ratio < 1:
        num_synth = int(max_samples * synth_ratio)
        num_real = max_samples - num_synth
        real_sample = real_df.sample(n=min(num_real, len(real_df)), random_state=42)
        synth_sample = synth_df.sample(n=min(num_synth, len(synth_df)), random_state=42)
        combined_df = pd.concat([real_sample, synth_sample], ignore_index=True)
    # only real data
    elif real_df is not None or synth_ratio == 0:
        combined_df = real_df.sample(n=min(max_samples, len(real_df)), random_state=42)
    # only synthetic data
    elif synth_df is not None or synth_ratio == 1:
        combined_df = synth_df.sample(
            n=min(max_samples, len(synth_df)), random_state=42
        )
    else:
        raise ValueError("At least one dataset must be provided.")

    return combined_df.sample(frac=1, random_state=42).reset_index(drop=True)


def preprocess_data(df, label2id):
    """Tokenize the text and add labels."""

    def tokenize_function(examples):
        encoding = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            verbose=False,
        )
        encoding["labels"] = [label2id[label] for label in examples["label"]]
        return encoding

    return df.map(tokenize_function, batched=True, remove_columns=df.column_names)


def compute_metrics(p):
    # Single-label multi-class classification
    y_pred = np.argmax(p.predictions, axis=1)
    y_true = p.label_ids

    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def train_model(
    train_df, dev_df, labels, output_dir, id2label=None, label2id=None, save_model=True
):
    """Train the model"""

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
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
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

    ### GET THE TABLE THAT IS SHOWED DURING TRAINING (EPOCH, LOSS, ACCURACY, ETC per epoch)
    # extrac the train history
    log_history = trainer.state.log_history
    log_df = pd.DataFrame(log_history)
    # get what we are interested on
    loss_column = log_df[log_df["loss"].notnull()]["loss"]
    log_df = log_df[log_df["epoch"].notnull() & log_df["eval_f1_micro"].notnull()]
    log_df = log_df[
        [
            "epoch",
            "train_loss",
            "eval_loss",
            "eval_accuracy",
            "eval_f1_micro",
            "eval_f1_macro",
        ]
    ]
    log_df["train_loss"] = loss_column.values
    # aave to CSV in the experiment folder
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_df.to_csv(log_file_path, index=False)

    print(f"💾 experiment saved to: {output_dir}\n")

    if save_model is False:
        # delete checkpoints after training
        for checkpoint in [
            f for f in os.listdir(output_dir) if f.startswith("checkpoint-")
        ]:
            shutil.rmtree(os.path.join(output_dir, checkpoint))
        print(f"🗑️ save_model={save_model}: Checkpoints deleted after training.\n")

    return trainer.model  # return the best model


def evaluation(test_dataset, model_input):
    """
    Evaluates a model using tokenized input data.

    Parameters:
    - test_dataset: Dataset containing 'input_ids' and 'attention_mask' (and 'labels' if available).
    - modelname: Path to the trained model checkpoint.

    Returns:
    - Dictionary of evaluation metrics.
    """
    # load model
    if isinstance(model_input, str):
        # load from huggingface or local path
        model = AutoModelForSequenceClassification.from_pretrained(model_input)
    else:
        # assume it's already a model instance
        model = model_input
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    # eval
    with torch.no_grad():
        for batch in test_dataset:
            # test dataset is assumed already preprocessed
            input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0).to(device)
            attention_mask = (
                torch.tensor(batch["attention_mask"]).unsqueeze(0).to(device)
            )
            labels = batch["labels"]

            # get output + argmax
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.append(labels)

    # compute metrics
    eval_metrics = {
        "f1_micro": f1_score(all_labels, all_preds, average="micro"),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "accuracy": accuracy_score(all_labels, all_preds),
    }

    display_metrics = {k: f"{v:.4f}" for k, v in eval_metrics.items()}
    print("🚀 Metrics on dev set:", display_metrics)
    return eval_metrics


def save_log(details, log_file):
    """Save details of all experiments in a json file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(details)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)


def main_multiclassRoBERTA(
    real_df=None,
    synth_df=None,
    dev_df=None,
    synth_ratio=0.0,
    max_samples=500,
    output_dir="experiment_output",
    log_dir="log.json",
    generation_method=None,  # generic vs targeted augmentation
    save_model=True,
    save_dataset=True,
):
    """
    Train a multi-class RoBERTA model using real and synthetic datasets.

    STEPS:
    1. Combine real and synthetic datasets.
    2. Tokenize and preprocess the data.
    3. Train the model.
    4. Evaluate
    5. Save the experiment details.
    """
    set_seed(42)

    if dev_df is None:
        raise ValueError("A development dataset (dev_df) must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    # combine datasets
    combined_df = combine_datasets(real_df, synth_df, synth_ratio, max_samples)

    # save datasets for reproducibility
    train_path = os.path.join(output_dir, "train.csv")
    dev_path = os.path.join(output_dir, "dev.csv")
    if save_dataset:
        # save the actual train and dev sets
        combined_df.to_csv(train_path, index=False)
        dev_df.to_csv(dev_path, index=False)

    # preprocess data
    labels = combined_df["label"].unique().tolist()
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    train_dataset = preprocess_data(Dataset.from_pandas(combined_df), label2id)
    dev_dataset = preprocess_data(Dataset.from_pandas(dev_df), label2id)

    # train
    start_time = time.time()
    labels_ids = [label2id[label] for label in labels]
    best_model = train_model(
        train_dataset,
        dev_dataset,
        labels_ids,
        output_dir,
        id2label,
        label2id,
        save_model,
    )
    end_time = time.time()

    # evaluate
    eval_metrics = evaluation(dev_dataset, best_model)

    if synth_df is None:
        synth_ratio = 0.0
    if real_df is None:
        synth_ratio = 1.0

    train_details = {
        "experiment_name": os.path.basename(output_dir),
        "experiment_dir": output_dir,
        "generation_method": generation_method,
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "train_size": len(combined_df),
        "dev_size": len(dev_df),
        "synthetic_ratio": synth_ratio,
        "train_time_seconds": end_time - start_time,
        "metrics_dev": eval_metrics,
    }

    # save log
    save_log(train_details, log_dir)

    return train_details
