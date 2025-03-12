import torch
import sentencepiece
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import csv
import gc
import os
import random

gc.collect()
torch.cuda.empty_cache()

device = 'cuda:0'

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large",torch_dtype=torch.float16,device_map='auto').to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def conv(txt):
    txt = txt.replace('[','').replace(']','').replace("'","")
    if ', ' in txt:
        txtL = txt.split(', ')
    else:
        txtL = [txt]
    return txtL

def gettest(fname):
    labels = []
    df = pd.read_csv(fname, sep=',', skiprows=1)
    df = df.dropna().reset_index(drop=True)
    data = df.values.tolist()
    result = []
    for row in data:

        result.append([row[0], row[1], row[2].lower()])
        if row[2].lower() not in labels:
            labels.append(row[2].lower())

    return result,labels


def gettestMultilabel(fname):
    labels = []
    df = pd.read_csv(fname, sep=',', skiprows=1)
    df = df.dropna().reset_index(drop=True)
    data = df.values.tolist()
    result = []
    for row in data:
        txtL = conv(row[2])
        for l in txtL:
            l = l.replace('"','')

            if l.lower() not in labels:
                labels.append(l.lower())
        result.append([row[0], row[1], row[2]])


    return result,labels

def getTrain(fname):
    df = pd.read_csv(fname, sep=',', skiprows=1)
    df = df.dropna().reset_index(drop=True)
    dataperlabel = {}
    for row in df.values.tolist():
        if len(row[1].split()) > 100:  # 80for newsgroups;ohsumed
            txtl = row[1].split()[:100]
            txt = ' '.join(t for t in txtl)
        else:
            txt = row[1]
        #txt = row[1]
        label = row[2].replace('[','').replace(']','').replace('"','').replace("'","").lower()
        try:
            dataperlabel[label].append(txt)
        except KeyError:
            dataperlabel[label] = [txt]

    selected = []
    for l in dataperlabel.keys():
        selected.append([random.sample(dataperlabel[l],1)[0],l])


    return selected



def prompt(selected,labels,instruction):
    options = ", ".join(l for l in labels)
    instruction = instruction+".The options are - {}.\n".format(options)
    elements = []
    for i in range(0, len(selected)):
        el = 'Text:'+selected[i][1]+'. Category:'+selected[i][2]

        elements.append(el)

    fewshot = instruction + "\n".join(item for item in elements)


    return fewshot



def runFewShot(fewshot,testdata,fnameres):
    with open(fnameres, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        for row in testdata:
            id = row[0]
            txt = row[1]
            actual = row[2]
            prompt = fewshot + txt + "\n\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs)
            predicted = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            writer.writerow([id, txt, actual, predicted.lower()])



def main():
    #testdata, labels = gettest('')
    testdata, labels = gettestMultilabel('')
    instruction = ''

    selected = getTrain('')
    pd.DataFrame(selected).to_csv(''+str(i)+'.csv')
    fnameres = ''+str(i)+'.csv'

    fewshot = prompt(selected, labels,instruction)
    runFewShot(fewshot,testdata,fnameres)


main()









