import string, re
import numpy as np
import pandas as pd
import midi
import mido
from mido import MidiFile
from mido.sockets import PortServer, connect

INSTRUMENTS = ['Acoustic Grand Piano','Bright Acoustic Piano','Electric Grand Piano','Honky-tonk Piano','Electric Piano 1','Electric Piano 2','Harpsichord','Clavi','Celesta','Glockenspiel','Music Box','Vibraphone','Marimba','Xylophone','Tubular Bells','Dulcimer','Drawbar Organ','Percussive Organ','Rock Organ','Church Organ','Reed Organ','Accordion','Harmonica','Tango Accordion','Acoustic Guitar (nylon)','Acoustic Guitar (steel)','Electric Guitar (jazz)','Electric Guitar (clean)','Electric Guitar (muted)','Overdriven Guitar','Distortion Guitar','Guitar harmonics','Acoustic Bass','Electric Bass (finger)','Electric Bass (pick)','Fretless Bass','Slap Bass 1','Slap Bass 2','Synth Bass 1','Synth Bass 2','Violin','Viola','Cello','Contrabass','Tremolo Strings','Pizzicato Strings','Orchestral Harp','Timpani','String Ensemble 1','String Ensemble 2','SynthStrings 1','SynthStrings 2','Choir Aahs','Voice Oohs','Synth Voice','Orchestra Hit','Trumpet','Trombone','Tuba','Muted Trumpet','French Horn','Brass Section','SynthBrass 1','SynthBrass 2','Soprano Sax','Alto Sax','Tenor Sax','Baritone Sax','Oboe','English Horn','Bassoon','Clarinet','Piccolo','Flute','Recorder','Pan Flute','Blown Bottle','Shakuhachi','Whistle','Ocarina','Lead 1 (square)','Lead 2 (sawtooth)','Lead 3 (calliope)','Lead 4 (chiff)','Lead 5 (charang)','Lead 6 (voice)','Lead 7 (fifths)','Lead 8 (bass + lead)','Pad 1 (new age)','Pad 2 (warm)','Pad 3 (polysynth)','Pad 4 (choir)','Pad 5 (bowed)','Pad 6 (metallic)','Pad 7 (halo)','Pad 8 (sweep)','FX 1 (rain)','FX 2 (soundtrack)','FX 3 (crystal)','FX 4 (atmosphere)','FX 5 (brightness)','FX 6 (goblins)','FX 7 (echoes)','FX 8 (sci-fi)','Sitar','Banjo','Shamisen','Koto','Kalimba','Bag pipe','Fiddle','Shanai','Tinkle Bell','Agogo','Steel Drums','Woodblock','Taiko Drum','Melodic Tom','Synth Drum','Reverse Cymbal','Guitar Fret Noise','Breath Noise','Seashore','Bird Tweet','Telephone Ring','Helicopter','Applause','Gunshot']
NOTES = ['do', 'dod', 're', 'mib', 'mi', 'fa', 'fad', 'sol', 'lab', 'la', 'sib', 'si']
OCTAVES = list(range(11))
OCTAVE = [',,,,,',',,,,',',,,',',,',',','',"'","''","'''","''''"]
NOTES_IN_OCTAVE = len(NOTES)

errors = {
    'program': 'Bad input, please refer this spec-\n'
               'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/program_change.htm',
    'notes': 'Bad input, please refer this spec-\n'
             'http://www.electronics.dit.ie/staff/tscarff/Music_technology/midi/midi_note_numbers_for_octaves.htm'
}

def order_chord(chordL,stem_octave=False):
    if stem_octave:
        replace = {",,":"",",":"","'":"","''":""}
        rem_octave = re.compile("(%s)" % "|".join(map(re.escape, replace.keys())))
        chordL = [rem_octave.sub(lambda mo: replace[mo.group()],x) for x in chordL]
    try:
        pL = [NOTES.index(x) for x in chordL]
    except:
        pL = list(range(len(chordL)))
    return np.argsort(pL)

def ordered_chord(chordL,stem_octave=False):
    if stem_octave:
        replace = {",,":"",",":"","'":"","''":""}
        rem_octave = re.compile("(%s)" % "|".join(map(re.escape, replace.keys())))
        chordL = [rem_octave.sub(lambda mo: replace[mo.group()],x) for x in chordL]

    pL = [NOTES.index(x) for x in chordL]
    pL = [chordL[x] for x in np.argsort(pL)]
    return pL

def iterate_notes():
    """ returns a list of notes times octaves"""
    l = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
    l = ['do','dod','re','mib','mi','fa','fad','sol','lab','la','sib','si']
    n = [str(x) for x in range(0,10)]
    n = [',,,,,',',,,,',',,,',',,',',','',"'","''","'''","''''"]
    s = []
    for j in n:
        for i in l:
            s.append(i+j)
    return s

def iterate_chords():
    """return the dict of major chords"""
    l = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    n = NOTES + NOTES
    c = {"M":[0,4,7],"m":[0,3,7],"sus":[0,5,7],"dim":[0,4,6],"m7":[0,3,7,9]}
    m = list(c.keys())
    s, s1 = {}, {}
    for i, l1 in enumerate(l):
        for m1 in m:
            k = l1 + ":" + m1
            l2 = l.index(l1)
            c1 = "-".join([n[x + l2] for x in c[m1]])
            s[k] = c1
            s1[c1] = k
    return s, s1

def iterate_scales():
    """return the dict of chords for each scale"""
    l = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    c = {"I":[0,4,7],"II":[2,5,9],"III":[4,7,11],"IV":[0,5,9],"V":[2,7,11],"VI":[0,4,9],"VII":[2,5,11]}
    n = NOTES + NOTES
    m = list(c.keys())
    s, s1 = {}, {}
    for i, l1 in enumerate(l):
        for m1 in m:
            k = l1 + ":" + m1
            l2 = l.index(l1)
            c1 = "-".join([n[x + l2] for x in c[m1]])
            s[k] = c1
            s1[c1] = k
    return s, s1

def iterate_scale(tonic='C'):
    """return the dict of chords for a given scale"""
    l = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    if not tonic in l:
        raise(ValueError,"tonic note not recognized")
    
    c = {"I":[0,4,7],"II":[2,5,9],"III":[4,7,11],"IV":[0,5,9],"V":[2,7,11],"VI":[0,4,9],"VII":[2,5,11]}
    n = NOTES + NOTES
    m = list(c.keys())
    s, s1 = {}, {}
    for m1 in m:
        k = tonic + ":" + m1
        l2 = l.index(tonic)
        c1 = "-".join([n[x + l2] for x in c[m1]])
        s[k] = c1
        s1[c1] = k
    return s, s1


def instrument_to_program(instrument: str) -> int:
    """return program code for instrument"""
    assert instrument in INSTRUMENTS, errors['program']
    return INSTRUMENTS.index(instrument) + 1

def program_to_instrument(program: int) ->  str:
    """return program for instrument"""
    assert 1 <= program <= 128, errors['program']
    return INSTRUMENTS[program - 1]

def number_to_note(number: int) -> tuple:
    """return the midi number of a note"""
    octave = number // NOTES_IN_OCTAVE
    assert octave in OCTAVES, errors['notes']
    assert 0 <= number <= 127, errors['notes']
    note = NOTES[number % NOTES_IN_OCTAVE]
    pitch = note + OCTAVE[octave]
    return pitch

def note_to_number(note: str, octave: int) -> int:
    """return the note for a midi number"""
    assert note in NOTES, errors['notes']
    assert octave in OCTAVES, errors['notes']
    note = NOTES.index(note)
    note += (NOTES_IN_OCTAVE * octave)
    assert 0 <= note <= 127, errors['notes']
    return note

def msg2dict(msg):
    """convert a mido message to a dict"""
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def switch_note(last_state, note, velocity, on_=True):
    """update state for a note"""
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    """update current note state"""
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]

def track2seq(track,mid):
    """convert midi tracks to lilypond style"""
    chorD, chorDr = iterate_scales()
    tempo = mid.ticks_per_beat
    pitchOn, pitchOff, durD = {}, {}, {}
    currT = 0
    for t in track:
        pitchOn[currT], pitchOff[currT] = [], []
        currT += t.time

    currT = 0
    for t in track:
        currT += t.time
        if hasattr(t,'note'):
            on = t.velocity > 0 and t.type == 'note_on'
            pitch = number_to_note(t.note)
            if on == False:
                pitchOff[currT].append(pitch) 
            else:
                pitchOn[currT].append(pitch)
                
    timeL = list(pitchOn.keys())
    deltaL = []
    for x,y in zip(timeL[1:],timeL[:-1]):
        deltat = (tempo/(x-y)*4)
        s = '%.0f' % deltat
        if deltat >= 1.6 and deltat < 1.7:
            s = '2'
        elif deltat >= 5.3 and deltat < 5.4:
            s = '8.'
        elif deltat >= 2.6 and deltat < 2.7:
            s = '4.'
        deltaL.append(s)

    deltaL.append("")
    last_time, tick_tot, tick_n = 0, 0, 0
    pitchD = {}
    colN = "lead"
    for i, currT in enumerate(pitchOff):
        if len(pitchOn[last_time]) == 0:
            pitchOn[last_time] = 'r1'
        pO = pitchOn[last_time]
        pL = order_chord(pO,stem_octave=True)
        pL = [pO[x]+deltaL[i] for x in pL]
        lilyS = " ".join(pL)
        if len(pL) > 1:
            lilyS = "<<" + " ".join(pL) + ">>"
            if len(pL) > 2:
                colN = "chord"
        try:
            pitchD[tick_n] += " " + lilyS
        except:
            pitchD[tick_n] = lilyS            
        delta_t = currT - last_time
        tick_tot += (delta_t/tempo/4)
        tick_n = (int) (tick_tot)
        last_time = currT

    resD = pd.DataFrame.from_dict(pitchD,orient="index",columns=[colN])
    if resD[colN].apply(lambda x: bool(re.search(",",x))).sum() > 20: # guessing the baseline
        resD.rename(columns={colN:"bass"},inplace=True)
    return resD
            
def song2array(midiF):
    mid = MidiFile(midiF, clip=True)
    tracks_len = [len(tr) for tr in mid.tracks]
    head = mid.tracks[0]
    signature = ''
    tempo = 0
    metaD = {}
    for h in head:
        if h.type == "time_signature":
            metaD['signature'] = "%d/%d" % (h.numerator,h.denominator)
        if h.type == "set_tempo":
            metaD['tempo'] = h.tempo
    songL = []*(len(mid.tracks)-1)
    for tr in mid.tracks[1:]:
        trL = track2seq(tr,mid)
        songL.append(trL)
    songD = pd.concat(songL,axis=1)
    return songD, metaD

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

def writeLy(melS,accS,harmS,keyS="sol \\major"):
    keyS = "\n\\key sol \\major\n"
    headerS = '\\version "2.12.3"\n\\include "italiano.ly"\ntitle = "Gen"\ndate="14-11-2023"\nserial ="2"\n\\include "Header.ly"\n'
    bodyS = '\\score{\n <<\n\\new ChordNames{\n\\tempo  4 = 100\n\\time 4/4\n\\set chordChanges = ##t\n\\armonia\n}\n'
    bodyS += '\\new Staff{\n\\set Staff.midiInstrument = #"violin"\n\\set Staff.instrument = "Violino"\n\\mark "andante" \n\\clef treble\n\\key ' + keyS + "\n\\time 4/4\n\\set Score.timeSignatureFraction = #'(1 . 4)\n\\set Staff.beatGrouping = #'(2 2 2 2)\n\\set Score.dynamicAbsoluteVolumeFunction = #Dinamica\n\\override Beam #'auto-knee-gap = #4\n\\set Score.beatLength = #(ly:make-moment 1 8)\n\melodia\n}\n"
    bodyS += '\\new Staff{\n\\time 4/4\n\\clef bass\n\\set Staff.midiInstrument = #"cello"\n\\set Staff.instrument = "Basso"\n\\set Score.dynamicAbsoluteVolumeFunction = #Dinamica\n\\key ' + keyS + '\n\\accompagnamento\n}\n'
    bodyS += '\n>>\n'
    bodyS += '\\midi{\n\\context {\n \\Score tempoWholesPerMinute = #(ly:make-moment 70 4)\n}\n}\n'
    bodyS += '\\layout{papersize = "a4"\npagenumber = "yes"\n\\context{\\Staff\n\\remove "Time_signature_engraver"\n}\n}\n}\n'
    footerS = ''
    fileS = headerS
    fileS += "melodia = {\n<<\n\\transpose do do' {\n" + melS + "\n}\n>>\n}\n"
    fileS += "armonia = {\n<<\n {\n" + harmS + "\n}\n>>\n}\n"
    fileS += "accompagnamento = {\n<<\n {\n" + accS + "\n}\n>>\n}\n"
    fileS += bodyS
    fileS += footerS
    return fileS

def process_midi(midiF):
    import music21
    try:
        midi = music21.converter.parse(midiF)
        parts = music21.instrument.partitionByInstrument(midi)
        partL, noteL = {}, []
        for par in range(len(parts)):
            for element_by_offset in music21.stream.iterator.OffsetIterator(parts[p]):
                for entry in element_by_offset:
                    if entry.isNote:
                        p = entry.pitch
                        n = p.italian + OCTAVE[p.octave] + str(int(4/entry.duration.quarterLength))
                        noteL.append(str(entry.pitch))
                    elif entry.isChord:
                        noteL.append('.'.join(str(n) for n in entry.normalOrder))
                    elif entry.isRest:
                        noteL.append('r')
            partL[par] = noteL
        return partL
    except Exception as e:
        print("failed on ", path)
        pass


def ly2array(midiF): #TOFIX
  import ly
  from ly.cli.main import load
  from ly.cli import command
  import ly.pitch
  import ly.duration
  import ly.dom
  import ly.music.read
  import itertools
  doc = ly.document.Document.load(midiF)
  doc = load(midiF,'UTF-8',None)
  cursor = ly.document.Cursor(doc)
  docR = ly.music.read.Reader(source)
  docR.consume()
  musR = ly.music.items.Document(doc)


  
  cursor.start_block
  octave = ''
  language = ly.docinfo.DocInfo(cursor.document).language() 
  r = ly.pitch.pitchReader(language)("do")
  p1 = ly.pitch.Pitch(*r, octave=ly.pitch.octaveToNum(octave))
  r = ly.pitch.pitchReader(language)("re")
  p2 = ly.pitch.Pitch(*r, octave=ly.pitch.octaveToNum(octave))
  version = ly.docinfo.DocInfo(cursor.document).version()
  absolute = False
  transposer = ly.pitch.transpose.Transposer(p1, p2)
  ly.pitch.transpose.transpose(cursor, transposer, language, absolute)  
  start = cursor.start
  cursor.start = 0

  
  
  depth = source.state.depth()
  pitches = ly.pitch.PitchIterator(source, language)
  psource = pitches.pitches()      
  lastPitch = ly.pitch.Pitch.c1()
  relPitch = [] # we use a list so it can be changed from inside functions
  with cursor.document as d:
    for t in itertools.chain(source, (None,)):
      print(t)
      #t = next(psource)
      if isinstance(t, ly.pitch.Pitch) or isinstance(t, ly.lex.lilypond.Rest):
        print(t)
        pass
      elif isinstance(t, ly.lex.lilypond.ChordMode):
        print(t)
      elif isinstance(t, ly.dom.Staff):
        print(t)
    
    lastPitch = t
    if in_selection(t):
      relPitch.append(lastPitch)
  while True:
    # eat stuff like \new Staff == "bla" \new Voice \notes etc.
    if isinstance(source.state.parser(), ly.lex.lilypond.ParseTranslator):
      t = consume()
    elif isinstance(t, ly.lex.lilypond.NoteMode):
      t = next(tsource)
    else:
      break
  
  for t in tsource:
    yield t

