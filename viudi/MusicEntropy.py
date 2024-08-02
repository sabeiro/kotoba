#!/usr/bin/python
import sys
import midi
from EHisto import *
import logging            

class MusicEntropy(object): 
    BassNote = 33
    NRange = 60
    NTrack = 8
    NRhyt = 10
    NChrom = NRange*NTrack
    NTot = NRange*NTrack*NRhyt
    chromProb = EHisto(12)
    histoScale = EHisto(12)
    label = [0]*NRange
    IsEvol = 0 #to write the entropy evolution file
    IsAnalized = 0
    EvolTime = 3000 #ms before dumping a entropy evolution information
    stream = 0
    def __init__(self,midifile):
        logging.warning("Processing: " + midifile)
        self.stream = midi.read_midifile(midifile)
#stream.textdump()
        tempo = float(self.stream.tempomap[0].tempo)
        self.sedicesimi = tempo/(31.*60.)
        ChromProb = [1.*1,16.*15.,9.*8.,6.*5.,5.*4.,4.*3.,45.*32.,3.*2.,8.*5.,5.*3.,9.*5.,15.*8.,2.*1.]
        ChromProb = [1.,1000.,1.,10000.,1.,1.,10000.,1.,10000.,1.,1000.,1.]
        #ChromProb = [1.,1000.,5.,10000.,3.,4.,10000.,2.,10000.,6.,1000.,7.]        
        Count = 0.
        for i in range(12):
            Count += 1./ChromProb[i]
            for i in range(12):
                self.chromProb.set(i,1./(ChromProb[i]*Count))
        for i in range(0,self.NRange):
            idx = i + self.BassNote
            self.label[i] = self.stream.NoteMap(idx)
    def __str__(self):
        string = "NTrack %d NoteRange %d NRhyt %d " %(self.NTrack,self.NoteRange,self.NRhyt)
        return string
    # def __setattr__(self):
    #     print "Nothing to change"
    # def __getattr__(self):
    #     print "Nothing to see"

    def ProcFile(self):
        state = [0]*self.NChrom
        histoTot = EHisto(self.NTot)
        histoRhyt = EHisto(self.NRhyt)
        histoChrom = EHisto(self.NChrom)
        FEvol = open("EntropyEvolution.dat","w")
        Count = 0.
        CurrTime = 0
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
        print ERhyt, EChrom, ETot

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


