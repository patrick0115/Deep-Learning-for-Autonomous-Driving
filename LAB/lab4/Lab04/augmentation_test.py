import torchaudio
from scipy.io.wavfile import write
import random
from os import listdir
from os.path import isfile, isdir, join
import matplotlib.pyplot as plt
import soundfile as sf
import os
import librosa
import librosa.display
# mypath = "./speech_commands/yes"
# mypath = "./speech_commands/up"
# mypath = "./speech_commands/down"

mypath = "./speech_commands/stop"
files = listdir(mypath)

# print(files)
# print(os.path.join(mypath,files[1]))
for i in range(10):
    audio, sr = sf.read(os.path.join(mypath,files[i]))
    print(mypath,files[i])
    # print(type(audio))
    print(audio.shape)
    print(sr)
audio, sr = sf.read(os.path.join(mypath,files[0]))
audio1, sr1= sf.read(os.path.join(mypath,files[1]))



plt.figure()
plt.subplot(2, 1, 2)
librosa.display.waveshow(audio.astype('float'), sr=sr)
plt.title('original')
# plt.xlim(0.8358,1.1)
plt.subplot(2, 1, 1)
# librosa.display.waveshow(audio1.astype('float'), sr=sr)
plt.plot(audio) 
plt.title('original')
# plt.xlim(0.835,1.1)
plt.show()