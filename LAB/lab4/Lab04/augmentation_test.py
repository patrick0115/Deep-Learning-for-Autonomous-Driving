import torchaudio
from scipy.io.wavfile import write
import random
from os import listdir
from os.path import isfile, isdir, join



mypath = "./speech_commands/no"
mypath = "./speech_commands/no"
mypath = "./speech_commands/no"
mypath = "./speech_commands/no"
mypath = "./speech_commands/no"
mypath = "./speech_commands/no"
mypath = "./speech_commands/stop"
mypath = "./speech_commands/yes"
# mypath = "./speech_commands/up"
# mypath = "./speech_commands/down"

files = listdir(mypath)

for f in files:
    

    fullpath = join(mypath, f)
    waveform1, sample_rate1 = torchaudio.load(fullpath)
    if isfile(fullpath):
        print(sample_rate1)




# a=random.randrange(1, 10)
# if a==1:
#     effects = [
#         ["lowpass", "300"],  
#         ["speed", "0.8"], 
#         ['pitch', '250'],
#     ]
# elif a==2:
#         effects = [
#         ["lowpass", "300"],  
#         ['pitch', '250'],
#     ]
# elif a==3:
#         effects = [
#         ["lowpass", "300"],  
#         ['pitch', '200'],
#     ]
# elif a==4:
#         effects = [
#         ["lowpass", "300"],  
#         ["speed", "1.2"],
#     ]
# elif a==5:
#         effects = [
#         ["lowpass", "300"],  
#         ["speed", "1.1"],
#     ]
# elif a==6:
#         effects = [
#         ["lowpass", "300"],  
#         ["speed", "1.1"], 
#         ['pitch', '-150'],
#     ]
# elif a==7:
#         effects = [
#         ["lowpass", "300"],  
#         ["speed", "1.2"], 
#         ['pitch', '-200'],
#     ]
# elif a==8:
#         effects = [
#         ["lowpass", "1000"],  
#     ]
# else:
#     effects = [
#             ["speed", "1"],  
#     ]

# waveform_aug, sample_rate_aug = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
# return waveform_aug, sample_rate_aug


