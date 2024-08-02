import math
import sys
#import numpy as np
#from scipy import linalg as la
#import matplotlib.pyplot as plt

datfile = sys.argv[1]
f = open(datfile,"r") 
NEntry = 0
Level = []
Freq = []
LevMin = 0
LevMax = 0
pMax = 0
for line in f: # read rest of lines
    line1 = []
    line1.append([(x) for x in line.split()])
    x = float(line1[0][0])
    y = float(line1[0][1])
    Freq.append(x)
    Level.append(y)
    if(LevMin < y): LevMin = y
    if(LevMax < y): LevMax = y
    NEntry += 1
f.close()

PLevel = []
PFreq = []
Max = LevMin
NSteep = 4
NPeak = 0
for i in range(NSteep,NEntry-NSteep):
    IsContinue = 1
    if(Max < Level[i]): Max = Level[i]
    for j in range(0,NSteep-1):
        if( Level[i-j-1] > Level[i-j]):
            IsContinue = 0
        if(Level[i+j+1] > Level[i+j]):
            IsContinue = 0
    #print IsContinue
    #print Level[i-3],Level[i-2], Level[i-1],Level[i],Level[i+1],Level[i+2],Level[i+3]
    if(IsContinue):
        PFreq.append(Freq[i])
        PLevel.append(Level[i])
        Max = LevMin
        NPeak += 1

LevMax = PLevel[0]
for i in range(NPeak):
    if(LevMax < PLevel[i]): 
        LevMax = PLevel[i]
        pMax = i

print "Max freq %.0f, level %.2f, NPeak %d" %(PFreq[pMax],LevMax,NPeak)
datfile = datfile.replace(".dat","Conv.dat")
print datfile
fOut = open(datfile,"w") 
for i in range(NPeak):
    vol = pow(2.,(-LevMax+PLevel[i])/10.)
    string = "%.0f %.2f %.5f\n" %(PFreq[i],PLevel[i],vol)
    fOut.write(string)
fOut.close()
