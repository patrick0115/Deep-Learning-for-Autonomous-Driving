from scipy.io.wavfile import write
import random
import librosa
sampling_rate=16000

def pitch(data, sampling_rate, pitch_factor):
    
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

def aug(waveform):
    a=random.randrange(1, 10)
    if a==1:
        waveform=pitch(waveform,sampling_rate, 1.5)
    elif a==2:
        waveform=speed(waveform, 1.2)
    elif a==3:
        waveform=pitch(waveform,sampling_rate, -1.5)
 
    elif a==4:
        waveform=speed(waveform, 0.85)
    elif a==5:
        waveform=pitch(waveform,sampling_rate, 1.5)
        waveform=speed(waveform, 1.2)
    elif a==6:
        waveform=pitch(waveform,sampling_rate, -1.5)
        waveform=speed(waveform, 0.85)
    elif a==7:
        waveform=pitch(waveform,sampling_rate, 1.5)
        waveform=speed(waveform, 0.85)
    elif a==8:
        waveform=speed(waveform, 1.2)
        waveform=pitch(waveform,sampling_rate, -1.5)
    else:
        waveform=waveform
    return waveform




