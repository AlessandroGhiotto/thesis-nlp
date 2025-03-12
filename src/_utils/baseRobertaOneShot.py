import pandas as pd
from datasets import load_dataset,Dataset,DatasetDict
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import f1_score

device = 'cuda:0'

tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")
#f1 = evaluate.load("f1")
f1 = evaluate.load('f1', average='macro')

def mappinglabels(mappingfile):
    #id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    #label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    id2label = {}
    label2id = {}

    with open(mappingfile) as f:
        lines = f.readlines()

    for line in lines:
        labelid = int(line.split('\t')[0])
        label = line.split('\t')[1].strip()
        id2label[labelid] = label
        label2id[label] = labelid

    return id2label, label2id



def preprocess_data(trainf,devf):

    train = load_dataset("csv", data_files=trainf,encoding='utf-8')
    dev = load_dataset("csv", data_files=devf,encoding='utf-8')
    tokenized_train = train.map(tokenizedata, batched=True)
    tokenized_dev = dev.map(tokenizedata, batched=True)


    return tokenized_train,tokenized_dev



def tokenizedata(examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average='macro')
    #return accuracy.compute(predictions=predictions, references=labels)



def classification(tokenized_train,tokenized_dev,id2label,label2id,modelname,numlabels):
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=numlabels, id2label=id2label, label2id=label2id).to(device)

    training_args = TrainingArguments(
        output_dir=modelname,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        #metric_for_best_model='f1',
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train['train'],
        eval_dataset=tokenized_dev['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )
    trainer.train()



def evaluation(test,label2id,modelname):
    data = load_dataset("csv", data_files=test)
    task_evaluator = evaluator("text-classification")
    eval_results = task_evaluator.compute(
        model_or_pipeline=modelname,
        data=data['train'],
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        label_mapping=label2id
    )
    return eval_results



def evaluationMulticlass(test,id2label, label2id,modelname):
    predictedLabels = []
    trueLabels = []
    model = AutoModelForSequenceClassification.from_pretrained(modelname)
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=True)
    testd = load_dataset("csv", data_files=test, encoding='utf-8')

    for el in testd['train']:

        encoded_input = tokenizer(el['text'], return_tensors='pt',truncation=True)
        output = model(**encoded_input)
        predicted_label = output.logits.argmax(-1).detach().numpy()[0]
        actualL = el['label']
        predictedLabels.append(predicted_label)
        trueLabels.append(actualL)

    #fscore = f1_score(trueLabels, predictedLabels)
    macroF1 = f1_score(trueLabels,predictedLabels,average='macro')
    microF1 = f1_score(trueLabels, predictedLabels, average='micro')

    return macroF1, microF1
    #return fscore

def main():
    train = ""
    dev = ""
    test = ""
    mappingfile = ""
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    id2label, label2id = mappinglabels(mappingfile)
    modelname = ""

    tokenized_train,tokenized_dev = preprocess_data(train,dev)

    classification(tokenized_train,tokenized_dev, id2label, label2id,modelname,len(id2label))


main()


