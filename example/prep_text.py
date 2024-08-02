import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, string
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.clean_text as c_t
import kotoba.text_stat as t_s
import importlib

if False: ## anima
    baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
    textD = pd.read_csv(baseDir + "anima0.csv",sep="\t",header=None,names=["day","time","person","message"])
    def convert_day(x):
        try:
            y = pd.to_date(x, format='%d/%M/%Y')
        except:
            y = np.nan
        return y
    textD['day'] = textD['day'].apply(lambda x: convert_day(x))
    textD1 = pd.read_csv(baseDir + "anima2.csv",sep="\t",header=None,names=["day","time","person","message"])
    def convert_day(x):
        try:
            y = pd.to_date(x, format='%m/%d/%Y')
        except:
            y = np.nan
        return y
    textD1['day'] = textD1['day'].apply(lambda x: convert_day(x))
    #text = c_t.preproc_wazup(textD[2])
    
    #text.to_csv(baseDir+"anima"+".txt",index=False,header=False,doublequote=False,quoting=False,escapechar="\\") #,quotechar=""

if False: ## scritti
    baseDir = os.environ['HOME'] + "/lav/siti/scritti/markdown/"
    baseDir = os.environ['HOME'] + "/lav/src/spiega/markdown/"
    fileS = ""
    for f in os.listdir(baseDir):
        with open(baseDir + f,"r") as fi:
            fileS += fi.read().rstrip()
            fi.close()

    fileL = fileS.splitlines()
    fileT = [x.split("\.") for x in fileL if x != '']
    fileG = [preproc_token(x[0]) for x in fileT]
    baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/raw/"
    with open(baseDir + "spiega.txt","w") as fi:
        fi.write("\n".join(fileG))
        fi.close()
    
if False: ## corrispondenza
    baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/Corrispondenza/csv/"
    mailL = []
    for f in os.listdir(baseDir):
        mailD = pd.read_csv(baseDir + f)
        mailD.columns = ['title','from','to','date','boh','subject']
        mailL.append(mailD)

    mailD = pd.concat(mailL)
    pat = c_t.split_pattern()
    fileL = mailD['subject'].apply(lambda x: re.split(pat,x))
    fileL = fileL.apply(lambda y: [preproc_token(x) for x in y if x != ''])
    from itertools import chain
    fileG = list(chain(*fileL))
    baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/raw/"
    with open(baseDir + "corrispondenza.txt","w") as fi:
        fi.write("\n".join(fileG))
        fi.close()
    # corpus = corpus.apply(lambda x: x.split("To:")[-1])
    # x = corpus[2]

if False: ## word frequency
    baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
    fName = baseDir + "anima" + ".txt"
    text = open(fName, encoding="utf-8").read()
    importlib.reload(c_t)
    text = c_t.clean_string(text)
    textL = text.split("\n")
    importlib.reload(t_s)
    corpus_freq = t_s.word_freq(textL)
    par = t_s.freq_power_law(corpus_freq,is_plot=True)
    
    

