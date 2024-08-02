import os, sys, re, random
os.environ['LAV_DIR'] = '/home/sabeiro/lav/'
dL = os.listdir(os.environ['LAV_DIR']+'/src/')
sys.path = list(set(sys.path + [os.environ['LAV_DIR']+'/src/'+x for x in dL]))
from viudi import pyMidi as p_m
from viudi import midi_etl as m_e
from viudi import midi_stream as m_s
import kotoba.transformer_translate as t_t
import kotoba.bert_transformer as b_t
import kotoba.clean_text as c_t
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib

opt = {"maxlen":80,"dropout_rate":0.1
      ,"baseDir":os.environ["HOME"]+"/lav/viudi/mus/","dir":["Bolla","Malastro"],"saveDir":os.environ["HOME"]+"/lav/kotoba/"
       ,"trans_dir":os.environ['HOME'] + '/lav/kotoba/raw/'
       ,"trans_file":os.environ['HOME'] + '/lav/kotoba/raw/chord_lead.csv'
       ,"prompt_file":os.environ['HOME'] + '/lav/kotoba/raw/alpaca_data.csv.gz'
       ,"vocab_size":100,"sequence_length":20,"batch_size":64
       ,"embed_dim":12,"latent_dim":24,"num_heads":8
       ,"max_decoded_sentence_length":20,"epochs":500
       }

if True: # translation data
  importlib.reload(c_t)
  train_pairs, val_pairs, test_pairs = c_t.parse_music_file(opt['trans_file'])

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
  tr_lead = t_t.trans_model(opt)
  tr_lead.fit(train_ds, epochs=opt['epochs'], validation_data=val_ds)
  t_t.save_model(tr_lead,opt['trans_dir'] + 'tr_lead')
  tr_bass = t_t.load_model(opt['trans_dir'] + 'tr_lead')
  
  importlib.reload(c_t)
  test_eng_texts = [pair[0] for pair in test_pairs]
  for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = c_t.decode_sequence(input_sentence,tr_lead,opt,eng_vect,spa_vect)
    print(input_sentence,translated)
 
if False: #create music
  importlib.reload(m_e)
  chorD, chorDs = m_e.iterate_scale(tonic="G")
  chorL = [chorD[x] for x in chorD]
  markov = pd.read_csv(opt['baseDir'] + "viurdi/markov_scale.csv")
  markov = markov[ (markov['author'] == "Bach")*(markov["mode"] == "major")]
  markov = markov[['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']].to_numpy()
  plt.imshow(markov)
  plt.show()
  l = list(range(7))
  markovL = markov.sum(axis=0)/markov.sum()
  start = np.random.choice(l, p=markovL)
  chordL = [chorL[start]]
  for i in range(60):
    p = markov[start]
    j = np.random.choice(l, p=markovL)
    chordL.append(chorL[j])
    start = j
    
  importlib.reload(c_t)
  harmS, melS, accS = '', '', ''
  for c in chordL:
    c = "-".join(sorted(c.split("-")))
    c = random.choice(test_eng_texts)
    notes = c_t.decode_sequence(c,tr_lead,opt,eng_vect,spa_vect)
    notes = re.sub("\[end\]","",re.sub("\[start\]","",notes))
    notes = re.sub("0","8",notes)
    notes = re.sub("fadla","fad la",notes)
    if notes == " ":
      continue
    length = [1/float(x) for x in re.findall(r'\d+',notes)]
    if sum(length) < 1:
      notes += "r%.0f" % (1/(1-sum(length)))
    length = [1/float(x) for x in re.findall(r'\d+',notes)]
    if int(sum(length)) != 1:
      continue
    chord = '<<' + re.sub("-"," ",re.sub("-","1 ",c,1)) + '>>'
    print(chord,notes)
    harmS += chord + "|\n"
    melS += notes + "|\n"

  with open(opt['baseDir'] + "viurdi/gen.ly", "w") as f:
    fileS = m_e.writeLy(
    f.write(fileS)
  

    
if False: # preprocess music
  fL = []
  for d in opt["dir"]:
    for f in os.listdir(opt["baseDir"] + d):
      fL.append(os.path.join(opt["baseDir"] + d, f))
      
  fL = [x for x in fL if re.search(".midi",x)]
  midiF = fL[2]
  importlib.reload(m_e)
  first_file = True
  for f in fL:
    songD, metaD = m_e.song2array(midiF)
    if not 'chord' in songD.columns:
      continue
    if not 'bass' in songD.columns:
      songD.loc[:,"bass"] = ""
    if first_file:
      songD[['chord','lead','bass']].to_csv(opt['trans_dir']+'chord_lead.csv', index=False)
      first_file = False
    else:
      songD[['chord','lead','bass']].to_csv(opt['trans_dir']+'chord_lead.csv', mode='a', header=False, index=False)

  importlib.reload(m_e)
  partL = m_e.process_midi(midiF)

# importlib.reload(m_s)
# pattern = m_s.read_midifile(midiF)



#from midiutil.MidiFile import MIDIFile
pL = mido.get_output_names()
port = mido.open_output(pL[0])

importlib.reload(p_m)
noteL = p_m.MidiFile(fL[0])result_array = m_e.mid2arry(mid)

importlib.reload(ly)


importlib.reload(m_e)
noteL = m_e.mid2arry(mid)
plotL = noteL
plt.plot(range(plotL.shape[0]), np.multiply(np.where(plotL>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
plt.title("nocturne_27_2_(c)inoue.mid")
plt.show()


  
for t in track:
  port.send(t)

port.panic()


#midi = p_m.MidiFile(fL[0])
	
				

  
header = tL[0]
print(header)  
  
tempo = mid.ticks_per_beat
time = 0
noteL = []*128

if False:
  # import pyaudio
  # p = pyaudio.PyAudio()
  import pygame
  clock = pygame.time.Clock()
  pygame.mixer.init(44100, -16, 2, 1024)
  pygame.mixer.music.set_volume(0.8)
  pygame.mixer.music.load(fL[0])
  pygame.mixer.music.play()


if False:
  for event in self.stream.iterevents():
    if isinstance(event, midi.NoteEvent):
      n = (event.pitch-self.BassNote)
      ns = n%12
      if (n < 0 or n >= self.NRange):
        string = "n out of range 0 < %d < %d, pitch %d" % (n,self.NRange,event.pitch)
        logging.warning(string)
        continue
      s = n*self.NTrack + event.track
      if (s < 0 or s >= self.NChrom):
        string = "s out of range 0 < %d < %d, track %d" % (s,self.NChrom,event.track)
        logging.warning(string)
        continue
      IsOn =  (0 != (event.statusmsg & 0x7f))
      IsDown = (event.velocity != 0)
      if(IsOn*IsDown):
        state[s] = event.msdelay
      else:
        dt = event.msdelay - state[s]
        if(dt <= 0): continue
        state[s] = 0
        q = int(math.log(dt*self.sedicesimi,2)) - 1
        if (q < 0 or q >= self.NRhyt):
          string = "q out of range 0 < %d < %d, dt %.0f" % (q,self.NRhyt,dt)
          logging.warning(string)
          continue
        s1 = s*self.NRhyt + q
        histoTot.add(s1,1.)
        histoChrom.add(s,1.)
        self.histoScale.add(ns,1.)
        histoRhyt.add(q,1.)
        Count += 1.
        CurrTime += event.msdelay
        if(self.IsEvol):
          if(CurrTime > self.EvolTime):
            ERhyt = histoRhtm.EntCalc(Count)
            EChrom = histoChrom.EntCalc(Count)
            ETot = histoTot.EntCalc(Count)
            FEvol.write("%d %.4f %.4f %.4f\n" % (event.msdelay/1000,ERhyt,EChrom,ETot))
            CurrTime = 0

    FEvol.close()  
    ERhyt = histoRhyt.EntCalc(Count)
    EChrom = histoChrom.EntCalc(Count)
    ETot = histoTot.EntCalc(Count)
    self.histoScale.Normalize()
    self.IsAnalized = 1
    print(ERhyt, EChrom, ETot)
    
  def Tonality(self):
    if(self.IsAnalized==0): self.ProcFile()
    corr = EHisto(12)
    for i in range(12):
      dist = self.histoScale.DistMat(self.chromProb)
      corr.set(i,dist)
      print self.label[i] + " dist: %.4f" % dist
      print self.chromProb
      self.chromProb.Rotate()
    n = corr.Min() 
    print self.label[0:12]
    print self.histoScale
    print corr
    print "Tonality: " + self.label[n]
    
    





from melodia.core.track import Track
from melodia.music import chord
from melodia.io import midi
from mido import MidiFile

track = Track(signature=(4, 4))

track.add(chord.maj('C3', (1, 1)))
track.add(chord.maj('D3', (1, 1)))
track.add(chord.min('A3', (1, 1)))
track.add(chord.maj7('G3', (1, 1)))

mid = MidiFile('VampireKillerCV1.mid', clip=True)
print(mid)


with open('chords.mid', 'wb') as f:
    midi.dump(track, f)

