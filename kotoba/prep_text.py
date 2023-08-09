import numpy as np
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import warnings; warnings.simplefilter('ignore')
import nltk
import string
from nltk import ngrams
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('tagsets')
nltk.download('averaged_perceptron_tagger')

def preproc_waz(text):
    text = text.str.strip()
    text = text.str.replace("\t", "")
    text = text.str.replace('"', "")
    text = text.str.replace("^ ", "")
    text = text.str.replace("<Media omitted>", "")
    text = text.str.replace("https", "")
    text = text[~text.isna()]
    printable = list(set(string.printable)) + ['à','è','ò','ù','é','ó',"ö","ä","ü"]
    text = text.apply(lambda s: ''.join(filter(lambda x: x in printable, s)) )
    setL = text == ''
    text = text[~setL]
    return text

def preproc_token(text):
  text = text.lower()  # Lowercase text
  text = re.sub(f"[{re.escape(punctuation)}]"," ", text)  # Remove punctuation
  #text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
  text = re.sub(r"https?://\S+", "", text)
  text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", text)
  text = re.sub(r"\b[0-9]+\b\s*", "", text)
  #text = " ".join([w for w in text.split() if not w.isdigit()])
  #text = " ".join([w for w in text.split() if w.isalpha()])
  #text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
  token = text.split()
  clean = [t for t in token if not t in stopL]
  text = " ".join(clean)
  return text

## anima

baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
cName = "anima_raw"
fName = baseDir + cName + ".txt"
textD = pd.read_csv(fName,sep=":",header=None)
text = preproc_waz(textD[2])
text.to_csv(baseDir+"anima"+".txt",index=False,header=False,doublequote=False,quoting=False,escapechar="\\") #,quotechar=""

## scritti

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
    
## corrispondenza
        
baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/Corrispondenza/csv/"
mailL = []
for f in os.listdir(baseDir):
    mailD = pd.read_csv(baseDir + f)
    mailD.columns = ['title','from','to','date','boh','subject']
    mailL.append(mailD)

mailD = pd.concat(mailL)
p = re.compile(r'[^\s\.][^\.\n]+')
p = re.compile(r"(?<!^)\s*[.\n]+\s*(?!$)")
p = re.compile(r'(?=\S)[^.\n]+(?<=\S)')
fileL = mailD['subject'].apply(lambda x: re.split(p,x))
fileL = fileL.apply(lambda y: [preproc_token(x) for x in y if x != ''])
from itertools import chain
fileG = list(chain(*fileL))
baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/raw/"
with open(baseDir + "corrispondenza.txt","w") as fi:
    fi.write("\n".join(fileG))
    fi.close()



## 
    
corpus = corpus.apply(lambda x: x.split("To:")[-1])
x = corpus[2]
stoplist = 'for a of the and to in \n \r'.split(' ')
stoplist = stoplist + ['den','und','die','-','uhr','che','un','il','per','di','e','la','>','>>','>>>']
stoplist = set(stoplist)
sentence = [[word for word in dc.lower().split() if word not in stoplist] for doc in corpus]
from collections import defaultdict
frequency = defaultdict(int)
for text in sentence:
    for token in text:
        frequency[token] += 1




# text = open(fName, encoding="utf-8").read()
text = text.lower()
text = text.translate(str.maketrans("", "", punctuation))

df['length'] = list(map(lambda x: len(str(x).split()), df['review']))
df = df.drop_duplicates(subset=["condition" ,"review", "rating"]).reset_index(drop=True)
df.isnull().any()
drugC = df.drugName.value_counts()
drugC = drugC[drugC>=5]
df = df.loc[df['drugName'].isin(drugC.index),]
condC = df.condition.value_counts()
condC = condC[condC>=5]
df = df.loc[df['condition'].isin(condC.index),]
df = df.loc[df['usefulCount']>6,]
df.loc[:,'condition'] = df['condition'].apply(lambda x: 'unknown' if re.search("users found",x) else str(x).lower())
df.loc[:,'drugName'] = df['drugName'].apply(lambda x: str(x).lower())
df.loc[:,'review'] = df['review'].apply(lambda x: str(x).lower())
df.review = df.review.str.lower()
df["condition"].fillna("unknown", axis=0, inplace=True)

df["condition"].nunique()
df["review"] = df.review.str.replace('"', "")
df["review"] = df.review.str.replace('&#039;', "")
df.review = df.review.str.replace(r'[^\x00-\x7F]+',' ')
#df.review = df.review.str.replace(r'^\s+|\s+?$','')
df.review = df.review.str.replace(r'\s+',' ')
df.review = df.review.str.replace(r'\.{2,}', '')
df.review = df.review.str.replace(r'\d+', ' ')
df.review = df.review.str.replace(r"\s*'\s*\w*", ' ')
df.review = df.review.str.replace(r'\W+', ' ')
df.review = df.review.str.replace(r'\s+', ' ')
df.review = df.review.str.replace(r'^\s+|\s+?$', '')
df
stop_spec = ['taking','pain','effects','first','started','like','months','get','days','time','would','one','weeks','took','week','also','got','month']
stop_spec.extend(['day','years','life','went','year','hours','going','used','lbs','getting','try','use','make','say'])
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from sklearn.feature_extraction import text
stop = text.ENGLISH_STOP_WORDS
stop_words.extend(['im', 'ive', 'it', 'mg', 'quot'])
stop_words.extend(stop)
stop_words.extend(stop_spec)
stop_words = list(set(stop_words))
for i in range(len(stop_words)):
    stop_words[i] = re.sub("'","",stop_words[i])
pat = r'\b(?:{})\b'.format('|'.join(stop_words))
pat

df['review'] = df['review'].str.replace(pat, '')
df.review = df.review.str.replace(r'\W+', ' ')
reviews = []
corpus=[]
for review in df['review']:
    reviews.append(review)
    corpus.append(nltk.sent_tokenize(review))
corpus=[sent for sublist in corpus for sent in sublist]
wordfreq = {}
for sentence in corpus:
    words = sentence.split()
    #tokens = nltk.word_tokenize(sentence) # To get the words, it can be also done with sentence.split()
    for word in words:
        if ( word not in wordfreq.keys() ): ## first time appearnce in the sentence
            wordfreq[word] = 1 # We initialize the corresponding counter
        else: ## if the world is already existed in the dictionalry 
            wordfreq[word] += 1 # We increase the corresponding counter
wordfreq = dict(sorted(wordfreq.items(),key= lambda x:x[1],reverse=True))
print(wordfreq)

len(list(wordfreq.keys()))
# Keeping 30 most preq words
corpus_freq = [(wordfreq[key],key) for key in list(wordfreq.keys())]
corpus_freq = [(word[1],word[0]) for word in corpus_freq[:60]] 

from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
lem = WordNetLemmatizer()
corpus_freq = [(lem.lemmatize(word[0]),word[1]) for word in corpus_freq]
np.array(list(cols.keys()))

cols = {word[0]: [] for word in corpus_freq}
reviews = pd.DataFrame(cols)
reviews.columns

def review_inpector(sentence, words):
    # Initializing an empty dictionary of word frequencies for the corresponding review
    tokens = nltk.word_tokenize(sentence)
    col_freq = {col:0 for col in words}
    # Filling the dictionary with word frequencies in the review
    for token in tokens:
        if token in words:
            col_freq[token] += 1

    return col_freq
my_list = list(map(review_inpector, df['review'],[list(cols.keys())]*df.shape[0] ) )
my_list[:2]
reviews = pd.DataFrame(my_list)
reviews['rating'] = df['rating'].reset_index(drop=True)
reviews
