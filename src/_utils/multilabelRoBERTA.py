from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import evaluate
from evaluate import evaluator
from sklearn.metrics import f1_score

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch


device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def loaddata(trainf, devf):
    train = load_dataset("csv", data_files=trainf, encoding="utf-8")
    dev = load_dataset("csv", data_files=devf, encoding="utf-8")
    return train, dev


def mappinglabels(train):
    labels = [
        label
        for label in train["train"].features.keys()
        if label not in ["id", "txt", "typet"]
    ]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    return id2label, label2id, labels


def preprocess_data(train, dev):
    tokenized_train = train.map(
        tokenizedata, batched=True, remove_columns=train["train"].column_names
    )
    tokenized_dev = dev.map(
        tokenizedata, batched=True, remove_columns=dev["train"].column_names
    )
    return tokenized_train, tokenized_dev


def tokenizedata(examples):
    labels = []
    # take a batch of texts
    text = examples["txt"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["label"] = labels_matrix.tolist()

    return encoding


def classification(
    tokenized_train, tokenized_dev, id2label, label2id, modelname, numlabels
):

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-large",
        problem_type="multi_label_classification",
        num_labels=numlabels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=modelname,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_dev["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.2):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {
        "f1": f1_micro_average,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "f1-macro": f1_macro_average,
    }
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result


def predictMultilable(test):
    labels = ["identity_hate", "insult", "obscene", "severe_toxic", "threat", "toxic"]
    allpredictions = []
    allactual = []
    model = AutoModelForSequenceClassification.from_pretrained(
        "/home/aleks/PycharmProjects/pythonProject/RobertaFiles/toxicLarge/checkpoint-2216"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/aleks/PycharmProjects/pythonProject/RobertaFiles/toxicLarge/checkpoint-2216",
        use_fast=True,
    )
    testd = load_dataset("csv", data_files=test, encoding="utf-8")
    for el in testd["train"]:

        labels_batch = {k: el[k] for k in el.keys() if k in labels}

        labels_matrix = np.zeros((1, len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        actuallabel = np.array(labels_matrix.tolist()[0])

        encoding = tokenizer(el["txt"], return_tensors="pt", truncation=True)
        encoding = {k: v.to(model.device) for k, v in encoding.items()}

        outputs = model(**encoding)

        logits = outputs.logits

        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.2)] = 1
        allpredictions.append(predictions)
        allactual.append(actuallabel)

    print(len(allactual))
    print(len(allpredictions))
    f1_micro_average = f1_score(
        y_true=allactual, y_pred=allpredictions, average="micro"
    )
    f1_macro_average = f1_score(
        y_true=allactual, y_pred=allpredictions, average="macro"
    )

    # return as dictionary
    metrics = {"f1": f1_micro_average, "f1-macro": f1_macro_average}
    return metrics


def main():
    trainf = "/home/aleks/PycharmProjects/pythonProject/RobertaFiles/toxictrain.csv"
    devf = "/home/aleks/PycharmProjects/pythonProject/RobertaFiles/toxicdev.csv"
    testf = "/home/aleks/PycharmProjects/pythonProject/RobertaFiles/toxictest.csv"
    modelname = "/home/aleks/PycharmProjects/pythonProject/toxicLarge"

    train, dev = loaddata(trainf, devf)

    id2label, label2id, labels = mappinglabels(train)
    print(labels)
    tokenized_train, tokenized_dev = preprocess_data(train, dev)
    # classification(tokenized_train, tokenized_dev, id2label, label2id, modelname,len(labels))
    metrics = predictMultilable(testf)
    print(metrics)


if __name__ == "__main__":
    main()
