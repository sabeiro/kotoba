import os, sys, json, re
import mlflow
import pandas as pd
import numpy as np
import openai
import shap
import torch
import transformers
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
baseDir = os.environ['HOME'] + '/lav/dauvi/portfolio/audit/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.chatbot_utils as c_t
import kotoba.evaluation_metric as e_m
import importlib
matplotlib.use('Qt5Agg')

fL = [x for x in os.listdir(baseDir) if re.search("pred_",x)]
perfL = []
for i, f in enumerate(fL):
    modN = fL[i].split("_")[1].split(".")[0]
    ansDf = pd.read_csv(baseDir + fL[i])
    ansDf.loc[:,"ref_answer"] = ansDf["ref_answer"].apply(lambda x: False if x != "VERA" else True)
    ansDf.loc[:,"pred_answer"] = ansDf["pred_answer"].apply(lambda x: False if x != "True" else True)
    y_pred = pd.Series(ansDf["pred_answer"],dtype=bool)
    y_ref = pd.Series(ansDf['ref_answer'],dtype=bool)
    clasR = classification_report(y_ref,y_pred)
    confM = confusion_matrix(y_ref,y_pred)
    tp = confM[0][0] + confM[1][1]
    fa = confM[0][1] + confM[1][0]
    f1 = 2*tp/(2*tp+fa)
    perf = {"model":modN,"precision":tp/confM.sum(),"f1-score":f1}
    perfL.append(perf)

perfDf = pd.DataFrame(perfL)
perfDf.to_csv(baseDir + "performance_results.csv",index=False)

pmodel = transformers.pipeline("question-answering")
tokenized_qs = None  
def f(questions, tokenized_qs, start):
    outs = []
    for q in questions:
        idx = np.argwhere(np.array(tokenized_qs["input_ids"]) == pmodel.tokenizer.sep_token_id)[0, 0]  
        d = tokenized_qs.copy()
        d["input_ids"][:idx] = q[:idx]
        d["input_ids"][idx + 1 :] = q[idx + 1 :]
        out = pmodel.model.forward(**{k: torch.tensor(d[k]).reshape(1, -1) for k in d})
        logits = out.start_logits if start else out.end_logits
        outs.append(logits.reshape(-1).detach().numpy())
    return outs

def tokenize_data(data):
    for q in data:
        question, context = q.split("[SEP]")
        tokenized_data = pmodel.tokenizer(question, context)
    return tokenized_data  

def f_start(questions):
    return f(questions, tokenized_qs, True)

def f_end(questions):
    return f(questions, tokenized_qs, False)

def out_names(inputs):
    question, context = inputs.split("[SEP]")
    d = pmodel.tokenizer(question, context)
    return [pmodel.tokenizer.decode([id]) for id in d["input_ids"]]


f_start.output_names = out_names
f_end.output_names = out_names

data = ["What is on the table?[SEP]When I got home today I saw my cat on the table, and my frog on the floor."]  
tokenized_qs = tokenize_data(data)
explainer_start = shap.Explainer(f_start, shap.maskers.Text(tokenizer=pmodel.tokenizer, output_type="ids"))
shap_values_start = explainer_start(data)
shap.plots.text(shap_values_start)
explainer_end = shap.Explainer(f_end, pmodel.tokenizer)
shap_values_end = explainer_end(data)
shap.plots.text(shap_values_end)

def make_answer_scorer(answers):
    def f(questions):
        out = []
        for q in questions:
            question, context = q.split("[SEP]")
            results = pmodel(question, context, topk=20)
            values = []
            for answer in answers:
                value = 0
                for result in results:
                    if result["answer"] == answer:
                        value = result["score"]
                        break
                values.append(value)
            out.append(values)
        return out

    f.output_names = answers
    return f

f_answers = make_answer_scorer(["my cat", "cat", "my frog"])
explainer_answers = shap.Explainer(f_answers, pmodel.tokenizer)
shap_values_answers = explainer_answers(data)
shap.plots.text(shap_values_answers)




