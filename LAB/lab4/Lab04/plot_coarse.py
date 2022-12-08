from torch.utils.data import DataLoader
from speech_command_dataset import SpeechCommandDataset
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from model import M5
import matplotlib.pyplot as plt
import shutil
import os
from sklearn.metrics import confusion_matrix
import itertools
from os import listdir
from os.path import isfile, isdir, join
import random
import time 
from tqdm import tqdm

if __name__ == '__main__':
    best_train_acc=[]
    best_loss=[]
    best_test_acc=[]
    epoch=[]
    checkpoint = open('./log/best_model_clean_batchsize_512.txt', 'r')
    for line in checkpoint.read().splitlines():     
        print(line)
        a=line.index("train_acc")
        b=line.index("loss")
        c=line.index("test_acc")
        d=line.index("epoch")
        best_train_acc.append(float(line[a+10:a+15]))
        best_loss.append(float(line[b+5:b+9]))
        best_test_acc.append(float(line[c+9:c+14]))
        epoch.append(int(line[d+6:a-1]))
        
    xtrain_acc = epoch[np.argmax(best_test_acc)]
    ytrain_acc = np.amax(best_test_acc)
   
    

    checkpoint_fine = open('./log/coarse_1_6_01_46_batchsize_256.txt', 'r')
    for line in checkpoint_fine.read().splitlines():     
        # print(line)
        a=line.index("train accuracy")
        b=line.index("loss")
        c=line.index("test accuracy")
        d=line.index("Epoch")
        # print(line[a+16:a+21])
        # print(line[b+6:b+10])
        # print(line[c+15:c+20])
        # print(line[d+7:a-20])
        best_train_acc.append(float(line[a+16:a+21]))
        best_loss.append(float(line[b+6:b+10]))
        best_test_acc.append(float(line[c+15:c+20]))
        epoch.append(592+int(line[d+7:a-20]))

    # checkpoint_fine = open('./log/coarse_2_5_07_17_batchsize_512.txt', 'r')
    # for line in checkpoint_fine.read().splitlines():     
    #     # print(line)
    #     a=line.index("train accuracy")
    #     b=line.index("loss")
    #     c=line.index("test accuracy")
    #     d=line.index("Epoch")
    #     # print(line[a+16:a+21])
    #     # print(line[b+6:b+10])
    #     # print(line[c+15:c+20])
    #     # print(line[d+7:a-20])
    #     best_train_acc.append(float(line[a+16:a+21]))
    #     best_loss.append(float(line[b+6:b+10]))
    #     best_test_acc.append(float(line[c+15:c+20]))
    #     epoch.append(992+int(line[d+7:a-20]))

    xtrain_acc = epoch[np.argmax(best_test_acc[0:592])]
    ytrain_acc = np.amax(best_test_acc[0:592])

    xtrain_acc1 = epoch[np.argmax(best_test_acc[593:992])]+592
    ytrain_acc1 = np.amax(best_test_acc[593:992])

    # xtrain_acc2 = epoch[np.argmax(best_test_acc[993:1293])]+992
    # ytrain_acc2 = np.amax(best_test_acc[993:1293])

    text= "Accuracy={:.2f}".format(ytrain_acc)
    # text= "Accuracy={:.2f}".format(ytrain_acc3)
    # fig= plt.figure()

    a=plt.plot( epoch, best_train_acc, color='b', label="Training accuracy")
    b=plt.plot( epoch, best_test_acc, color='c', label="Testing accuracy")
    plt.tick_params(axis='y')
    plt.ylim([55, 98])
    plt.title("Accuracy of coarse-grained ", fontsize=30 , weight='bold')
    plt.ylabel("Accuracy",fontsize=25)
    plt.xlabel("Epoch",fontsize=25)
    # annot_max(xtrain_acc, ytrain_acc)
    plt.annotate("Accuracy="+str(ytrain_acc), xy=(xtrain_acc, ytrain_acc), xytext=(xtrain_acc+5, ytrain_acc+1),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    plt.annotate("Accuracy="+str(ytrain_acc1), xy=(xtrain_acc1, ytrain_acc1), xytext=(xtrain_acc1-2, ytrain_acc1+2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    # plt.annotate("Accuracy="+str(ytrain_acc2), xy=(xtrain_acc2, ytrain_acc2), xytext=(xtrain_acc2-2, ytrain_acc2+2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    plt.annotate("Accuracy="+str(best_test_acc[424]), xy=(593, best_test_acc[424]), xytext=(424+20, best_test_acc[424]+2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    # plt.annotate("Accuracy="+str(best_test_acc[824]), xy=(993, best_test_acc[824]), xytext=(824+20, best_test_acc[824]-2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')    
    plt.legend(shadow=True, fancybox=True)
    plt.show()
    