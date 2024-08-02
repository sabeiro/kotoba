import string, os, sys, re
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.transformer_translate as t_t
import kotoba.bert_transformer as b_t
import kotoba.clean_text as c_t
import kotoba.text_gen_lstm as t_g
import importlib
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# origin = "http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
# content = requests.get("http://www.gutenberg.org/cache/epub/11/pg11.txt").text
opt = {"trans_file":os.environ['HOME'] + '/lav/kotoba/raw/en_es.txt'
       ,"prompt_file":os.environ['HOME'] + '/lav/kotoba/raw/alpaca_data.csv.gz'
       ,"vocab_size":5000,"sequence_length":20,"batch_size":64
       ,"embed_dim":256,"latent_dim":2048,"num_heads":8
       ,"max_decoded_sentence_length":20,"epochs":1
}

if False: # prompt data
  importlib.reload(c_t)
  promptD = pd.read_csv(opt['prompt_file'],compression="gzip")
  promptD['in'] = promptD.apply(lambda x: str(x['instruction']) + ': "' + str(x['input']) + '"',axis=1 )
  promptD['output'] = promptD['output'].apply(lambda x: '[start] ' + str(x) + ' [end]')
  text_pairs = promptD[['in','output']].to_numpy()
  train_pairs, val_pairs, test_pairs = c_t.split_pairs(text_pairs)

if True: # translation data
  importlib.reload(c_t)
  train_pairs, val_pairs, test_pairs = c_t.parse_trans_file(opt['trans_file'])

if False: # conversation data
  baseDir = os.environ['HOME'] + "/lav/kotoba/scritti/pers/"
  sentence = pd.read_csv(baseDir + "anima.txt")['Ciccia'].map(preprocess_text)
  sentence.dropna().to_csv(baseDir + "anima_clean.txt",index=False)

if True: # preprocess data
  train_eng_text = [pair[0] for pair in train_pairs]
  train_spa_text = [pair[1] for pair in train_pairs]
  eng_vect = c_t.vectorize_text(train_eng_text,opt['vocab_size'],opt['sequence_length'])
  spa_vect = c_t.vectorize_text(train_spa_text,opt['vocab_size'],opt['sequence_length']+1)
  train_ds = c_t.make_dataset(train_pairs,opt,eng_vect,spa_vect)
  val_ds = c_t.make_dataset(val_pairs,opt,eng_vect,spa_vect)
  c_t.show_example(train_ds)

if True: # train and test transformer
  importlib.reload(t_t)
  transformer = t_t.trans_model(opt)
  transformer.fit(train_ds, epochs=opt['epochs'], validation_data=val_ds)

  importlib.reload(c_t)
  test_eng_texts = [pair[0] for pair in test_pairs]
  for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = c_t.decode_sequence(input_sentence,transformer,opt,eng_vect,spa_vect)
    print(input_sentence,translated)

if False: # LSTM
  baseDir = "/home/sabeiro/tmp/pers/"
  cName = "markdown"
  opt = {"sequence_length":100,"batch_size":128,"n_epoch":30,"baseDir":baseDir+"/text/","cName":cName,"fName":baseDir+"text/"+cName+".txt","isLoad":True,"genCoding":False}
  text = open(opt['fName'],encoding="utf-8").read()
  #opt['isLoad'] = False
  #gen.gen_coding(text)
  gen.load_vocab(opt['baseDir'] + "english_vocab.txt")
  importlib.reload(t_g)
  gen = t_g.text_gen(opt)
  gen.train(n_epoch=2,text=text)
  gen.save_model()
  text = gen.clean_text(text)
  text = re.sub("\n"," ",text)
  for i in range(10):
    n = int(random.uniform(0,len(text)))
    seed = text[n:n+100]
    generated = gen.gen(seed=seed,n_chars=150)
    print(seed + ' -> ' + generated)
    

if False: # BERT
  importlib.reload(b_t)
  eng_texts, spa_texts = zip(*train_pairs[:100])
  eng_texts = list(eng_texts)
  spa_texts = list(spa_texts)
  eng = eng_vect(eng_texts)
  spa = spa_vect(spa_texts)
  embed_es = b_t.PositionalEmbedding(vocab_size=opt['vocab_size'], d_model=opt['embed_dim'])
  embed_en = b_t.PositionalEmbedding(vocab_size=opt['vocab_size'], d_model=opt['embed_dim'])
  es_emb = embed_es(spa)
  en_emb = embed_en(eng)
  
  importlib.reload(b_t)
  att = b_t.BaseAttention(num_heads=2, key_dim=512)
  s0 = att(en_emb, es_emb)
  att = b_t.CrossAttention(num_heads=2, key_dim=512)
  s1 = att(en_emb, es_emb)
  att = b_t.GlobalSelfAttention(num_heads=2, key_dim=512)
  s2 = att(en_emb, es_emb)
  att = b_t.CausalSelfAttention(num_heads=2, key_dim=512)
  s3 = att(en_emb, es_emb)
  l = np.argmax(np.sum(eng,axis=1))
  fig, ax = plt.subplots(4,1)
  ax[0].set_title("base attention")
  ax[0].imshow(np.mean(s0,axis=0))
  ax[1].set_title("cross attention")
  ax[1].imshow(np.mean(s1,axis=0)-np.mean(s0,axis=0))
  ax[2].set_title("global self attention")
  ax[2].imshow(np.mean(s2,axis=0)-np.mean(s0,axis=0))
  ax[3].set_title("causal self attention")
  ax[3].imshow(np.mean(s3,axis=0)-np.mean(s0,axis=0))
  plt.show()

  importlib.reload(b_t)
  sample_ffn = b_t.FeedForward(opt['embed_dim'], 2048)
  ffn = sample_ffn(en_emb)
  sample_encoder = b_t.EncoderLayer(d_model=opt['embed_dim'], num_heads=8, dff=2048)
  enc = sample_encoder(en_emb)
  fig, ax = plt.subplots(2,1)
  ax[0].set_title("feed forward")
  ax[0].imshow(np.mean(ffn,axis=0))
  ax[1].set_title("encoder")
  ax[1].imshow(np.mean(enc,axis=0))
  plt.show()
  
  translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))
  print_translation(sentence, translated_text, ground_truth)

  head = 0
  attention_heads = tf.squeeze(attention_weights, 0)
  attention = attention_heads[head]
  attention.shape
  translator = b_t.Translator(tokenizers, transformer)
  sentence = 'este é um problema que temos que resolver.'
  ground_truth = 'this is a problem we have to solve .'
  
  def print_translation(sentence, tokens, ground_truth):
    print(f'{"Input:":15s}: {sentence}')
    print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
    print(f'{"Ground truth":15s}: {ground_truth}')

  transformer = b_t.Transformer(
    num_layers=num_layers,d_model=d_model,num_heads=num_heads,dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),target_vocab_size=tokenizers.en.get_vocab_size().numpy(),dropout_rate=dropout_rate)
  embed_pt = PositionalEmbedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
  embed_en = PositionalEmbedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)
  pt_emb = embed_pt(pt)
  en_emb = embed_en(en)

