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
num=40
for j in range(150):
    audios=[]
    
    # print(files)
    # print(os.path.join(mypath,files[1]))
    for i in range(num):
        # print(j*num+i)
        audio, sr = sf.read(os.path.join(mypath,files[j*num+i]))
        audios.append(audio)

    plt.subplots(figsize=(15, 15))
    for i in range(num):
       
        plt.subplot(num//2, 2, i+1)
        plt.plot(audios[i])
        plt.title(files[j*num+i])
        # plt.title(i+1)
   
    plt.show()

