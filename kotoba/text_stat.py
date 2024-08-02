import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re, string
from collections import Counter
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

def char_freq(sentence):
  """frequence of each word, returns a defaultdict"""
  frequency = defaultdict(int)
  for text in sentence:
    for token in text:
        frequency[token] += 1
  return frequency

def word_freq(corpus):
  """frequency of each word from corpus, returns a list of tuple in descending order"""
  wordfreq = {}
  for sentence in corpus:
    words = sentence.split()
    for word in words:
      if ( word not in wordfreq.keys() ): 
        wordfreq[word] = 1 
      else: 
        wordfreq[word] += 1 
  wordfreq = dict(sorted(wordfreq.items(),key= lambda x:x[1],reverse=True))
  print("total number of words %d" % len(list(wordfreq.keys())) )
  corpus_freq = [(key,wordfreq[key]) for key in list(wordfreq.keys())]
  return corpus_freq

def freq_power_law(corpus_freq,is_plot=False):
  """power law of frequency decay"""
  i = list(range(len(corpus_freq)))
  l = [x[1] for x in corpus_freq]
  par = np.polyfit(i,np.log(l), 1)
  if is_plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    line, = ax.plot(l, color='blue', lw=2)
    ax.set_yscale('log')
    ax.set_xlabel("cardinality")
    ax.set_ylabel("frequency")
    plt.show()
  return par

def clean_by_freq(corpus_freq,min_freq=5):
  """select words from starting frequencies"""
  #corpus_freq = [(word[1],word[0]) for word in corpus_freq[:60]] 
  #nltk.download('omw-1.4')
  lem = WordNetLemmatizer()
  corpus_freq = [(lem.lemmatize(word[0]),word[1]) for word in corpus_freq]
  cols = {word[0]: [] for word in corpus_freq}
  return cols
  
def create_corpus(text_list):
  """corpus from list of text"""
  sentences = []
  corpus = []
  for sentence in text_list:
    sentences.append(sentence)
    corpus.append(nltk.sent_tokenize(sentence))
  corpus = [sent for sublist in corpus for sent in sublist]
  return corpus

def count_in_sentence(sentence, words):
  """count how often words appear in sentence"""
  tokens = sentence.split()
  col_freq = {col:0 for col in words}
  for token in tokens:
    if token in words:
      col_freq[token] += 1

  return col_freq
