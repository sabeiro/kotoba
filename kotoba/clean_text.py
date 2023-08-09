import pandas as pd
import numpy as np
import re, os, random
import spacy
from matplotlib import pyplot as plt
from string import punctuation
import nltk
from nltk.corpus import stopwords
import string, os, sys
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

def parse_trans_file(trans_file):
  with open(trans_file) as f:
    lines = f.read().split("\n")[:-1]
    text_pairs = []
  for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))

  for _ in range(5):
    print(random.choice(text_pairs))

  random.shuffle(text_pairs)
  num_val_samples = int(0.15 * len(text_pairs))
  num_train_samples = len(text_pairs) - 2 * num_val_samples
  train_pairs = text_pairs[:num_train_samples]
  val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
  test_pairs = text_pairs[num_train_samples + num_val_samples :]

  print(f"{len(text_pairs)} total pairs")
  print(f"{len(train_pairs)} training pairs")
  print(f"{len(val_pairs)} validation pairs")
  print(f"{len(test_pairs)} test pairs")

  return train_pairs, val_pairs, test_pairs

def decode_sequence(input_sentence,transformer,opt,eng_vect,spa_vect):
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
  """
  Remove non text tokens
  """
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

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

def vectorize_text(text,vocab_size,sequence_length):
  txt_vect = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize=custom_standardization,
  )
  txt_vect.adapt(text)
  return txt_vect

def format_dataset(eng, spa):
  return ({"encoder_inputs": eng, "decoder_inputs": spa[:, :-1],}, spa[:, 1:])

def make_dataset(pairs,opt,eng_vect,spa_vect):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    eng = eng_vect(eng_texts)
    spa = spa_vect(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng, spa))
    dataset = dataset.batch(opt['batch_size'])
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()



# baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
# sentence = pd.read_csv(baseDir + "anima.txt")['Ciccia'].map(preprocess_text)
# sentence.dropna().to_csv(baseDir + "anima_clean.txt",index=False)
