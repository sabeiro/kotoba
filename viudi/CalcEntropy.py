#!/usr/bin/python
import sys
from MusicEntropy import *

midifile = sys.argv[1]
me = MusicEntropy(midifile)
me.ProcFile()
CorrMat = me.Tonality()

