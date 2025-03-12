import torch
import sentencepiece
from transformers import LlamaTokenizer,LlamaForCausalLM
import pandas as pd
import csv
import gc
import os
import random

device = 'cuda:0'

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',torch_dtype=torch.float16,use_auth_token='').to(device)
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',use_auth_token='')

def gettest(fname):
    labels = []
    df = pd.read_csv(fname, sep=',', skiprows=1)
    df = df.dropna().reset_index(drop=True)
    data = df.values.tolist()
    result = []
    for row in data:

        label = row[2].lower()
        if label == 'sci/tech':
            label = 'science/tech'
        else:
            label = label

        result.append([row[0], row[1].replace('"', ''), label])
        random.shuffle(result)
        if label not in labels:
            labels.append(label)

    return result, labels




def conv(txt):
    txt = txt.replace('[','').replace(']','').replace("'","")

    if ', ' in txt:
        txtL = txt.split(', ')
    else:
        txtL = [txt]
    return txtL

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
        random.shuffle(result)


    return result,labels

def getSelected(tname):
    df = pd.read_csv(tname, sep=',')
    data = df.values.tolist()
    newdata = []
    for row in data:
        newdata.append([row[0],row[1],row[2]])

    return newdata



def prompt(selected,labels,instruction):
    options = ", ".join(l for l in labels)
    instruction = instruction+".The options are - {}.\n".format(options)
    elements = []
    for i in range(0, len(selected)):
        el = 'Text:'+selected[i][1]+'. Category:'+selected[i][2]

        elements.append(el)

    fewshot = instruction + "\n".join(item for item in elements)


    return fewshot




def runFewShotTask(fewshot,testdata,fnameres,labels):
    list_of_lengths = (lambda x: [len(i) for i in x])(labels)
    print('len: ',list_of_lengths)
    with open(fnameres, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        for row in testdata:
            id = row[0]
            txt = row[1]
            actual = row[2]
            prompt = fewshot+'\n'+'Text: '+txt+'. Category:'
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=max(list_of_lengths))
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,output_scores=True)[0]

            predicted = output.split('\n')

            answer = predicted[max(loc for loc, val in enumerate(predicted) if 'Category:' in val and txt in val)].split('Category:')[1]
            writer.writerow([id, txt, actual, answer.lower()])

def main():
    filenames = []
    instructions = []
    for i in range(0,len(filenames)):
        fname = fnames[i].split(':')[0]
        tname = fnames[i].split(':')[1]
        instruction = instructions[i]
        if fname.endswith(''):
            testdata, labels = gettest(''+fname)
        else:
            testdata, labels = gettestMultilabel(''+fname)

        selected = getSelected(tname)


        fnameres = ''

        fewshotb = prompt(selected, labels,instruction)
        runFewShotTask(fewshotb, testdata, fnameres, labels)

main()