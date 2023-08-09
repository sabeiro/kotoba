import re, os, random
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, FastText, TfidfModel
from gensim import corpora
from sklearn.decomposition import IncrementalPCA    
from sklearn.manifold import TSNE                   
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pprint
from collections import defaultdict
random.seed(0)

baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
sentence = pd.read_csv(baseDir + "anima_clean.txt")['Ciccia'].apply(lambda x: x.split(" ")).array

baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/raw/"
f = "sito_scritti.txt"
f = "corrispondenza.txt"
f = "spiega.txt"
with open(baseDir + f,"r") as fi: fileS = fi.read().rstrip()
sentence = fileS.split("\n")
sentence = [x.split() for x in sentence]

class MyCorpus:
  """An iterator that yields sentences (lists of str)."""
  def __iter__(self):
    corpus_path = datapath('lee_background.cor')
    for line in open(corpus_path):
      yield utils.simple_preprocess(line)

frequency = defaultdict(int)
for text in sentence:
  for token in text:
    frequency[token] += 1

# sentence = [[token for token in text if frequency[token] > threshold] for text in sentence]

dictionary = corpora.Dictionary(sentence)
# corpus = [dictionary.doc2bow(text) for text in sentence]
# corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus) 

#model = Word2Vec(sentence,min_count=10,workers=4)
model = Word2Vec(sentence,vector_size=100,window=5,min_count=5,workers=4)
#model.train(sentence[:100],total_examples=model.corpus_count,total_words=20,epochs=20)

def reduce_dimensions(model,num_dim=2):
  vectors = np.asarray(model.wv.vectors)
  labels = np.asarray(model.wv.index_to_key)  
  tsne = TSNE(n_components=num_dim, random_state=0)
  vectors = tsne.fit_transform(vectors)
  return vectors, labels

def perfPCA(model):
  vocab = [x for x in model.wv.key_to_index]
  X = [model.wv.get_vector(x, norm=True) for x in vocab]
  pca = PCA(n_components=2)
  result = pca.fit_transform(X)
  return result[:,0], result[:, 1], vocab

coord, labels = reduce_dimensions(model)
x_vals, y_vals = coord[:,0], coord[:,1]
fig, ax = plt.subplots(figsize=(12, 12))
ax.scatter(x_vals, y_vals,alpha=0.1)
indices = list(range(len(labels)))
selected = random.sample(indices, min(200,len(indices)))
#selected = indices[1:-1:10]
texts = [plt.text(x_vals[i],y_vals[i],labels[i],alpha=0.4) for i in selected]
#texts = [plt.annotate(labels[i],(x_vals[i],y_vals[i]),alpha=0.4) for i in selected]
#from adjustText import adjust_text
#adjust_text(texts,axis=ax)
plt.axis('off')
plt.show()

####

coord, label = reduce_dimensions(model,num_dim=3)
word3d = pd.DataFrame(coord,columns=["x","y","z"])
word3d['label'] = label
word3d['freq'] = [frequency[x] for x in label]
baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/raw/"
word3d.to_csv(baseDir + "word3d_spiega.csv",index=False)

new_doc = "wenn wir uns sehen"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)


model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
model.wv.doesnt_match("breakfast cereal dinner lunch".split(" "))
model.wv.similarity('woman', 'man')


# bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# tfidf = TfidfModel(bow_corpus)
# print(tfidf[dictionary.doc2bow(new_doc.lower().split())])

# model.save('/tmp/mymodel')
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')

