import os, sys, re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
import kotoba.transformer_text as t_t
import importlib

opt = t_t.defOpt()
opt = {"vocab_size":5000,"maxlen":80,"embed_dim":256,"num_heads":2,"feed_forward_dim":256,"dropout_rate":0.1,"batch_size":128
       ,"baseDir":os.environ["HOME"]+"/lav/src/spiega/","dir":["markdown"],"saveDir":os.environ["HOME"]+"/lav/kotoba/"}

opt['baseDir'] = os.environ["HOME"]+"/lav/kotoba/raw/"
opt['dir'] = ['diagnosis']

fileL = []
for d in opt["dir"]:
  for f in os.listdir(opt["baseDir"] + d):
    fileL.append(os.path.join(opt["baseDir"] + d, f))

print(f"{len(fileL)} files")

diaD = pd.read_csv(opt['baseDir'] + opt['dir'][0] + fileL[0])



random.shuffle(fileL)
importlib.reload(t_t)
text_ds = t_t.line_dataset(fileL)
text_ds = text_ds.shuffle(buffer_size=256)
text_ds = text_ds.batch(opt["batch_size"])
tP = t_t.textPrep(opt,text_ds)
vocab = tP.get_vocab()
text_ds = text_ds.map(tP.prep_inputs)
text_ds = text_ds.prefetch(t_t.data_tune())

for x, y in text_ds:
  print(x)
  plt.imshow(x)
  plt.show()
  break

word_to_index = {}
for index, word in enumerate(vocab):
  word_to_index[word] = index

importlib.reload(t_t)
start_prompt = "the project consists in "
start_tokens = [word_to_index.get(_,1) for _ in start_prompt.split()]
num_tokens_generated = 40
text_gen_callback = t_t.textGenerator(opt,num_tokens_generated,start_tokens,vocab)
model = t_t.textModel(opt).create_model()
model.fit(text_ds,verbose=2,epochs=25,callbacks=[text_gen_callback])

save_path = opt["saveDir"] + "model/"
model.save_weights(save_path+"_gen.h5")
model_json = model.to_json()
with open(save_path+"_gen.json", "w") as json_file:
  json_file.write(model_json)

json_file = open(save_path+'_gen.json','r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
json_file.close()
loaded_model.load_weights(save_path+"_gen.h5" )
loaded_model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
print('model loaded')
