import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import pandas as pd
import csv
import gc
import os
import random

device = "cuda:0"

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, use_auth_token=""
).to(device)
tokenizer = LlamaTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", use_auth_token=""
)


def gettest(fname):
    """Load single-label test data from CSV"""
    labels = []
    df = pd.read_csv(fname, sep=",", skiprows=1)
    df = df.dropna().reset_index(drop=True)
    data = df.values.tolist()
    result = []
    for row in data:

        label = row[2].lower()
        if label == "sci/tech":
            label = "science/tech"  # rename
        else:
            label = label

        result.append([row[0], row[1].replace('"', ""), label])  # id, text, label
        random.shuffle(result)

        # add label to list of unique labels
        if label not in labels:
            labels.append(label)

    return result, labels


def conv(txt):
    """Convert string to list"""
    txt = txt.replace("[", "").replace("]", "").replace("'", "")

    if ", " in txt:
        txtL = txt.split(", ")
    else:
        txtL = [txt]
    return txtL


def gettestMultilabel(fname):
    """Load multi-label test data from CSV"""
    labels = []
    df = pd.read_csv(fname, sep=",", skiprows=1)
    df = df.dropna().reset_index(drop=True)
    data = df.values.tolist()
    result = []
    for row in data:
        txtL = conv(row[2])
        for l in txtL:
            l = l.replace('"', "")

            # add label to list of unique labels
            if l.lower() not in labels:
                labels.append(l.lower())
        result.append([row[0], row[1], row[2]])
        random.shuffle(result)

    return result, labels


def getSelected(tname):
    """Load selected examples for few-shot learning"""
    df = pd.read_csv(tname, sep=",")
    data = df.values.tolist()
    newdata = []
    for row in data:
        newdata.append([row[0], row[1], row[2]])

    return newdata


def prompt(selected, labels, instruction):
    """Constructs the few-shot learning prompt"""
    # attach options to instruction
    options = ", ".join(l for l in labels)
    instruction = instruction + ".The options are - {}.\n".format(options)
    # create elements (text: <text>. Category: <category>)
    elements = []
    for i in range(0, len(selected)):
        el = "Text:" + selected[i][1] + ". Category:" + selected[i][2]
        elements.append(el)

    # attach elements to instruction
    fewshot = instruction + "\n".join(item for item in elements)
    return fewshot


def runFewShotTask(fewshot, testdata, fnameres, labels):
    """Run few-shot prompting task"""
    list_of_lengths = (lambda x: [len(i) for i in x])(labels)
    print("len: ", list_of_lengths)
    with open(fnameres, "a", encoding="UTF8") as f:
        writer = csv.writer(f)
        for row in testdata:
            id = row[0]
            txt = row[1]
            actual = row[2]
            # construct prompt
            prompt = fewshot + "\n" + "Text: " + txt + ". Category:"
            # generate output
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generate_ids = model.generate(
                inputs.input_ids, max_new_tokens=max(list_of_lengths)
            )
            output = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                output_scores=True,
            )[0]

            predicted = output.split("\n")

            # extract the predicted category
            answer = predicted[
                # Take the last of this occurrences
                max(
                    # extracts from "prediceted" occurrences of "Category:" and the original txt
                    loc
                    for loc, val in enumerate(predicted)
                    if "Category:" in val and txt in val
                )
                # extract the predicted category
            ].split("Category:")[1]
            writer.writerow([id, txt, actual, answer.lower()])


def main():
    filenames = []
    instructions = []
    for i in range(0, len(filenames)):
        fname = fnames[i].split(":")[0]
        tname = fnames[i].split(":")[1]
        instruction = instructions[i]
        if fname.endswith(""):
            testdata, labels = gettest("" + fname)
        else:
            testdata, labels = gettestMultilabel("" + fname)

        selected = getSelected(tname)

        fnameres = ""

        fewshotb = prompt(selected, labels, instruction)
        runFewShotTask(fewshotb, testdata, fnameres, labels)


main()
