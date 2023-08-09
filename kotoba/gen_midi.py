import os, sys, re
import midi
from mido import MidiFile
#from midiutil.MidiFile import MIDIFile

#conn = midi.MidiConnector('/dev/serial0')

opt = {"vocab_size":5000,"maxlen":80,"embed_dim":256,"num_heads":2,"feed_forward_dim":256,"dropout_rate":0.1,"batch_size":128
       ,"baseDir":os.environ["HOME"]+"/lav/viudi/mus/","dir":["Bolla","Malastro"],"saveDir":os.environ["HOME"]+"/lav/kotoba/"}

fL = []
for d in opt["dir"]:
  for f in os.listdir(opt["baseDir"] + d):
    fL.append(os.path.join(opt["baseDir"] + d, f))

fL = [x for x in fL if re.search(".midi",x)]
mid = MidiFile(fL[0], clip=True)
tL = []
for track in mid.tracks:
  tL.append(track)

for t in tL[0]:
  print(t)
  
tempo = mid.ticks_per_beat
time = 0
noteL = []*128


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

