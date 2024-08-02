import pandas as pd
import numpy as np
import re, os, random, csv
# import spacy
from matplotlib import pyplot as plt
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk import ngrams
from nltk.tokenize import word_tokenize 
from nltk.stem import SnowballStemmer
import string, os, sys
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('tagsets')
# nltk.download('averaged_perceptron_tagger')

def preproc_wazup(text):
  """preprocess whatsapp input"""
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

def preproc_text(text):
  """preprocess text"""
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

def clean_string(text):
  """clean string in a sentence"""
  text = text.lower()
  #text = text.translate(str.maketrans("", "", punctuation))
  stoplist = 'for a of the and to in \n \r'.split(' ')
  stoplist = stoplist + ['den','und','die','-','uhr','che','un','il','per','di','e','la','>','>>','>>>']
  stoplist = set(stoplist)
  textL = text.split(" ")
  sentence = [word for word in textL if word not in stoplist]
  sentence = " ".join(sentence)
  return sentence

def stopword_pattern():
  """returns a stopwords pattern"""
  stop_spec = ['taking','pain','effects','first','started','like','months','get','days','time','would','one','weeks','took','week','also','got','month']
  stop_spec.extend(['day','years','life','went','year','hours','going','used','lbs','getting','try','use','make','say'])
  stop_words = list(stopwords.words('english'))
  # from sklearn.feature_extraction import text
  # stop = text.ENGLISH_STOP_WORDS
  stop_words.extend(['im', 'ive', 'it', 'mg', 'quot'])
  stop_words.extend(stop)
  stop_words.extend(stop_spec)
  stop_words = list(set(stop_words))
  for i in range(len(stop_words)):
    stop_words[i] = re.sub("'","",stop_words[i])
  pat = r'\b(?:{})\b'.format('|'.join(stop_words))
  return pat

def split_pattern():
  """create a pattern for spliting sentences"""
  p = re.compile(r'[^\s\.][^\.\n]+')
  p = re.compile(r"(?<!^)\s*[.\n]+\s*(?!$)")
  p = re.compile(r'(?=\S)[^.\n]+(?<=\S)')
  return p

def clean_pandas(df,colname):
  """clean a pandas dataset with apply"""
  df['length'] = list(map(lambda x: len(str(x).split()), df[colname]))
  df.loc[:,colname] = df[colname].apply(lambda x: str(x).lower())
  df = df.drop_duplicates(subset=[colname]).reset_index(drop=True)
  df.isnull().any()
  # drugC = df.drugName.value_counts()
  # drugC = drugC[drugC>=5]
  # df = df.loc[df['drugName'].isin(drugC.index),]
  if False:
    condC = df.condition.value_counts()
    condC = condC[condC>=5]
    df = df.loc[df[colname].isin(condC.index),]
  df.loc[:,colname] = df[colname].apply(lambda x: 'unknown' if re.search("users found",x) else str(x).lower())
  df[colname].fillna("unknown", axis=0, inplace=True)
  df[colname].nunique()
  df[colname] = df[colname].str.replace('"', "")
  df[colname] = df[colname].str.replace('&#039;', "")
  df[colname] = df[colname].str.replace(r'[^\x00-\x7F]+',' ')
  df[colname] = df[colname].str.replace(r'^\s+|\s+?$','')
  df[colname] = df[colname].str.replace(r'\s+',' ')
  df[colname] = df[colname].str.replace(r'\.{2,}', '')
  df[colname] = df[colname].str.replace(r'\d+', ' ')
  df[colname] = df[colname].str.replace(r"\s*'\s*\w*", ' ')
  df[colname] = df[colname].str.replace(r'\W+', ' ')
  df[colname] = df[colname].str.replace(r'\s+', ' ')
  df[colname] = df[colname].str.replace(r'^\s+|\s+?$', '')
  pat = stopword_pattern()
  df[colname] = df[colname].str.replace(pat, '')
  df[colname] = df[colname].str.replace(r'\W+', ' ')
  return df

  
def open_trans_file(trans_file):
  """opens a translate file and create I/O pairs"""
  text_pairs = []
  with open(trans_file) as f:
    lines = f.read().split("\n")[:-1]
  for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))
  return text_pairs

def open_music_file(trans_file,rowL=[0,1]):
  """opens a translate file and create I/O pairs"""
  text_pairs = []
  with open(trans_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
          chord, note = row[rowL[0]], row[rowL[1]]
          chordL = re.findall("<<(.*?)>>", chord)
          c = " ".join([re.sub(" ","",x) for x in chordL])
          #c = " ".join([re.sub(" ","",x) for x in chordL[:1]])
          c = re.sub("[0-9]","-",c)
          c = re.sub("[,']","",c)
          c = c[:-1]
          c = "-".join(sorted(c.split("-")))
          n = "[start] " + note + " [end]"
          text_pairs.append((c,n))
  return text_pairs

def split_pairs(text_pairs,split_share=0.15):
  """split I/O pairs into train, test and validation"""
  random.shuffle(text_pairs)
  num_val_samples = int(split_share * len(text_pairs))
  num_train_samples = len(text_pairs) - 2 * num_val_samples
  train_pairs = text_pairs[:num_train_samples]
  val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
  test_pairs = text_pairs[num_train_samples + num_val_samples :]
  print("total: %d train: %d val: %d, test %d" % (len(text_pairs),len(train_pairs),len(val_pairs),len(test_pairs)))
  return train_pairs, val_pairs, test_pairs

def parse_trans_file(trans_file):
  """parse a translation file into training pairs"""
  text_pairs = open_trans_file(trans_file)
  for _ in range(5):
    print(random.choice(text_pairs))

  train_pairs, val_pairs, test_pairs = split_pairs(text_pairs)
  return train_pairs, val_pairs, test_pairs

def parse_music_file(trans_file,rowL=[0,1]):
  """parse a translation file into training pairs"""
  text_pairs = open_music_file(trans_file,rowL)
  for _ in range(5):
    print(random.choice(text_pairs))

  train_pairs, val_pairs, test_pairs = split_pairs(text_pairs)
  return train_pairs, val_pairs, test_pairs


def decode_sequence(input_sentence,transformer,opt,eng_vect,spa_vect):
  """decode model output into text"""
  spa_vocab = spa_vect.get_vocabulary()
  spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
  tokenized_input_sentence = eng_vect([input_sentence])
  decoded_sentence = "[start]"
  for i in range(opt['max_decoded_sentence_length']):
    tokenized_target_sentence = spa_vect([decoded_sentence])[:, :-1]
    predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
    sampled_token_index = np.argmax(predictions[0, i, :])
    sampled_token = spa_index_lookup[sampled_token_index]
    decoded_sentence += " " + sampled_token
    if sampled_token == "[end]":
      break
  return decoded_sentence

def preprocess_text(text):
  """  Remove non text tokens  """
  nltk.download('stopwords')
  #nlp = spacy.load("en_core_web_sm")
  stopL = stopwords.words('english') + stopwords.words('italian') + stopwords.words('german')
  text = text.lower()  # Lowercase text
  text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
  text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
  text = re.sub(r"https?://\S+", "", text)
  #text = re.sub(r"<a[^>]*>(.*?)</a>", r"\1", text)
  text = re.sub(r"\b[0-9]+\b\s*", "", text)
  text = " ".join([w for w in text.split() if not w.isdigit()])
  #text = " ".join([w for w in text.split() if w.isalpha()])
  #text = re.sub(r"[^A-Za-z0-9\s]+", "", text)
  token = text.split()
  clean = [t for t in token if not t in stopL]
  text = " ".join(clean)
  return text

def custom_standardization(input_string):
  """additional puctuation to strip"""
  strip_chars = string.punctuation + "¿"
  strip_chars = strip_chars.replace("[", "")
  strip_chars = strip_chars.replace("]", "")
  lowercase = tf.strings.lower(input_string)
  return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def vectorize_text(text,vocab_size,sequence_length):
  """vectorize text for model input"""
  txt_vect = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=custom_standardization,
  )
  txt_vect.adapt(text)
  return txt_vect

def format_dataset(eng, spa):
  """format dataset into [I,O],[O]"""
  return ({"encoder_inputs": eng, "decoder_inputs": spa[:, :-1],}, spa[:, 1:])

def make_dataset(pairs,opt,eng_vect,spa_vect):
  """takes I/O pairs, an option dict and two vectorize functions"""
  eng_texts, spa_texts = zip(*pairs)
  eng_texts = list(eng_texts)
  spa_texts = list(spa_texts)
  eng = eng_vect(eng_texts)
  spa = spa_vect(spa_texts)
  dataset = tf.data.Dataset.from_tensor_slices((eng, spa))
  dataset = dataset.batch(opt['batch_size'])
  dataset = dataset.map(format_dataset)
  return dataset.shuffle(2048).prefetch(16).cache()

def show_example(train_ds):
  """show few examples of training data"""
  for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

def split_chat(textL,partecipants=[]):
  """split text into partecipants"""
  outL = []
  Np = len(partecipants)
  for text in textL:
    sentenceP = ['']*Np
    for p in partecipants:
      if re.search(t,p):
        t = text.split(p)
        
