from scipy.io.wavfile import write
import random
import librosa
import numpy as np
sampling_rate=16000

def pitchup(data):
    return librosa.effects.pitch_shift(data, sampling_rate, 1.25)
def pitchdown(data):
    return librosa.effects.pitch_shift(data, sampling_rate, -1.25)
def speedup(data ):
    return librosa.effects.time_stretch(data, 1.15)
def speeddown(data ):
    return librosa.effects.time_stretch(data, 0.85)
def shift(samples):
    y_shift = samples.copy()
    timeshift_fac = 0.3*2*(np.random.uniform()-0.5)  
    start = int(y_shift.shape[0] * timeshift_fac)
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
    return y_shift
def value(samples):
  y_aug = samples.copy()
  dyn_change = np.random.uniform(low=2,high=2)
  return y_aug * dyn_change
def noise (samples,a):
  y_noise = samples.copy()
  noise_amp = a*np.random.uniform()*np.amax(y_noise)
  y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])
  return y_noise
def Streching(samples):
  input_length = len(samples)
  streching = samples.copy()
  streching = librosa.effects.time_stretch(streching.astype('float'), 1.02)
  if len(streching) > input_length:
      streching = streching[:input_length]
  else:
      streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
  return streching
  
def aug(waveform):
    a=random.randrange(1, 8)
    if a==1:
        waveform=noise(waveform,0.008)
        waveform=pitchup(waveform)
    elif a==2:
        waveform=noise(waveform,0.008)
        waveform=pitchdown(waveform)
    elif a==3:
        waveform=noise(waveform,0.008)
        waveform=speedup(waveform)
    elif a==4:
        waveform=noise(waveform,0.008)
        waveform=speeddown(waveform)
    elif a==5:
        waveform=noise(waveform,0.008)
        waveform=shift(waveform)
    elif a==6:
        waveform=noise(waveform,0.008)
        waveform=value(waveform)
    elif a==7:
        waveform=noise(waveform,0.02)
    else :
        waveform=noise(waveform,0.008)
        waveform=Streching(waveform)

    return waveform




