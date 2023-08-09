import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import string


def defOpt():
  return {"vocab_size":20000,"maxlen":80,"embed_dim":256,"num_heads":2,"feed_forward_dim":256
       ,"baseDir":"/home/sabeiro/tmp/","directories":["aclImdb/train/pos","aclImdb/train/neg","aclImdb/test/pos","aclImdb/test/neg"]}

def line_dataset(filenames):
  return tf.data.TextLineDataset(filenames)

def data_tune():
  return tf.data.AUTOTUNE

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
  """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
  """
  i = tf.range(n_dest)[:, None]
  j = tf.range(n_src)
  m = i >= j - n_src + n_dest
  mask = tf.cast(m, dtype)
  mask = tf.reshape(mask, [1, n_dest, n_src])
  mult = tf.concat(
    [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
  )
  return tf.tile(mask, mult)

class transformerBlock(layers.Layer):
  def __init__(self, opt):
    super().__init__()
    self.att = layers.MultiHeadAttention(opt["num_heads"], opt["embed_dim"])
    self.ffn = keras.Sequential(
      [layers.Dense(opt["feed_forward_dim"], activation="relu"), layers.Dense(opt["embed_dim"]),]
    )
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = layers.Dropout(opt["dropout_rate"])
    self.dropout2 = layers.Dropout(opt["dropout_rate"])

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
    attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
    attention_output = self.dropout1(attention_output)
    out1 = self.layernorm1(inputs + attention_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output)
    return self.layernorm2(out1 + ffn_output)

class tokenAndPositionEmbedding(layers.Layer):
  def __init__(self, opt):
    super().__init__()
    self.token_emb = layers.Embedding(input_dim=opt["vocab_size"], output_dim=opt["embed_dim"])
    self.pos_emb = layers.Embedding(input_dim=opt["maxlen"], output_dim=opt["embed_dim"])

  def call(self, x):
    maxlen = tf.shape(x)[-1]
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    x = self.token_emb(x)
    return x + positions

class textPrep():
  def __init__(self,opt,text):
    self.opt = opt
    def clean_text(input_string):
      """ Remove paths and handle punctuation """
      lowercased = tf.strings.lower(input_string)
      stripped = tf.strings.regex_replace(lowercased,"[^ ]*/[^ ]*", " ")
      return tf.strings.regex_replace(stripped, f"([{string.punctuation}])", r" \1")

    self.vect_lay = TextVectorization(standardize=clean_text,max_tokens=opt["vocab_size"]-1,
                                      output_mode="int",output_sequence_length=opt["maxlen"]+1)
    self.vect_lay.adapt(text)
    
  def prep_inputs(self,text):
    vocab_size = self.opt['vocab_size'] - 1
    text = tf.expand_dims(text, -1)
    tokenized_sentences = self.vect_lay(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y

  def get_vocab(self):      
    return self.vect_lay.get_vocabulary()
  
class textModel():
  def __init__(self,opt):
    self.opt = opt

  def create_model(self):
    inputs = layers.Input(shape=(self.opt["maxlen"],), dtype=tf.int32)
    embedding_layer = tokenAndPositionEmbedding(self.opt)
    x = embedding_layer(inputs)
    transformer_block = transformerBlock(self.opt)
    x = transformer_block(x)
    outputs = layers.Dense(self.opt["vocab_size"])(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # No loss and optimization based on word embeddings from transformer block
    model.compile("adam", loss=[loss_fn, None],)
    self.model = model
    return model

  def train(self,text):
    if not hasattr(self,'model'):
      self.create_model()
    self.model.fit(text,verbose=2,epochs=25)
  
  # def save_model(self):
    

  
class textGenerator(keras.callbacks.Callback):
  """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
  """
  
  def __init__(self,opt,max_tokens,start_tokens,index_to_word,top_k=10,print_every=1):
    self.max_tokens = max_tokens
    self.start_tokens = start_tokens
    self.index_to_word = index_to_word
    self.print_every = print_every
    self.opt = opt
    self.k = top_k

  def sample_from(self, logits):
    logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

  def detokenize(self, number):
    return self.index_to_word[number]

  def on_epoch_end(self, epoch, logs=None):
    maxlen = self.opt["maxlen"]
    start_tokens = [_ for _ in self.start_tokens]
    if (epoch + 1) % self.print_every != 0:
      return
    num_tokens_generated = 0
    tokens_generated = []
    while num_tokens_generated <= self.max_tokens:
      pad_len = maxlen - len(start_tokens)
      sample_index = len(start_tokens) - 1
      if pad_len < 0:
        x = start_tokens[:maxlen]
        sample_index = maxlen - 1
      elif pad_len > 0:
        x = start_tokens + [0] * pad_len
      else:
        x = start_tokens
        x = np.array([x])
        y, _ = self.model.predict(x)
        sample_token = self.sample_from(y[0][sample_index])
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        num_tokens_generated = len(tokens_generated)
        txt = " ".join(
          [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")

