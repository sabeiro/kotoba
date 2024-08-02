import librosa
import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Audio

filename = librosa.example('nutcracker')
y, sr = librosa.load(filename)
#Audio(data=y, rate=sr)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

hop_length = 512
y_harmonic, y_percussive = librosa.effects.hpss(y)
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfcc)
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),beat_frames)
chromagram = librosa.feature.chroma_cqt(y=y_harmonic,sr=sr)
beat_chroma = librosa.util.sync(chromagram,beat_frames,aggregate=np.median)
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

sr = 22050
y = librosa.chirp(fmin=32, fmax=32 * 2**5, sr=sr, duration=10, linear=True)
D = librosa.stft(y)
mag, phase = librosa.magphase(D)
freqs = librosa.fft_frequencies()
times = librosa.times_like(D)
phase_exp = 2*np.pi*np.multiply.outer(freqs,times)

fig, ax = plt.subplots()
img = librosa.display.specshow(np.diff(np.unwrap(np.angle(phase)-phase_exp, axis=1), axis=1, prepend=0),cmap='hsv',alpha=librosa.amplitude_to_db(mag, ref=np.max)/80 + 1,ax=ax,y_axis='log',x_axis='time')
ax.set_facecolor('#000')
cbar = fig.colorbar(img, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar.ax.set(yticklabels=['-π', '-π/2', "0", 'π/2', 'π']);
plt.show()


M = librosa.feature.melspectrogram(y=y)
M_highres = librosa.feature.melspectrogram(y=y, hop_length=512)
fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
librosa.display.specshow(librosa.power_to_db(M, ref=np.max),y_axis='mel', x_axis='time', ax=ax[0])
ax[0].set(title='44100/1024/4096')
ax[0].label_outer()
librosa.display.specshow(librosa.power_to_db(M_highres, ref=np.max),hop_length=512,y_axis='mel', x_axis='time', ax=ax[1])
ax[1].set(title='44100/512/4096')
ax[1].label_outer()
plt.show()



f0, voicing, voicing_probability = librosa.pyin(y=y, sr=sr, fmin=50, fmax=300)
S = np.abs(librosa.stft(y))
times = librosa.times_like(S, sr=sr)
fig, ax = plt.subplots()
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.plot(times, f0, linewidth=2, color='white', label='f0')
ax.legend()
plt.show()

harmonics = np.arange(1, 31)
frequencies = librosa.fft_frequencies(sr=sr)
harmonic_energy = librosa.f0_harmonics(S, f0=f0, harmonics=harmonics, freqs=frequencies)
# sphinx_gallery_thumbnail_number = 2
fig, ax = plt.subplots(nrows=2, sharex=True)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),y_axis='log', x_axis='time', ax=ax[0])
librosa.display.specshow(librosa.amplitude_to_db(harmonic_energy, ref=np.max),x_axis='time', ax=ax[1])
ax[0].label_outer()
ax[1].set(ylabel='Harmonics')
plt.show()
