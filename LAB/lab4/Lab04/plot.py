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
    print(xtrain_acc)
    print(ytrain_acc)
   


    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # a=ax1.plot( epoch, best_train_acc, color='b', label="Training accuracy")
    # b=ax1.plot( epoch, best_test_acc, color='c', label="Testing accuracy")
    # ax1.tick_params(axis='y')
    # ax1.set_ylim([65, 98])
    # ax1.set_title("Accuracy of best model without dirty data", fontsize=30 , weight='bold')
    # ax1.annotate(text, xy=(xtrain_acc, ytrain_acc), xytext=(xtrain_acc+15, ytrain_acc+0.5),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    # c=ax2.plot( epoch, best_loss, color='g',label="Loss")
    # ax2.tick_params(axis='y')
    # ax2.set_ylim([0.1, 0.5])
    # ax1.set_ylabel("Accuracy",fontsize=25)
    # ax2.set_ylabel("Loss",fontsize=25 )
    # ax1.set_xlabel("Epoch" ,fontsize=25)
    # ax1.legend(shadow=True, fancybox=True,loc=2)
    # ax2.legend(shadow=True, fancybox=True,loc=4)
    # plt.show()
    

    checkpoint_fine = open('./log/fine_grained_1_4_16_41_batchsize_256.txt', 'r')
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

    checkpoint_fine = open('./log/fine_grained_2_5_08_07_batchsize_256.txt', 'r')
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
        epoch.append(692+int(line[d+7:a-20]))


    checkpoint_fine = open('./log/fine_grained_3_7_08_54_batchsize_512.txt', 'r')
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
        epoch.append(792+int(line[d+7:a-20]))

    xtrain_acc = epoch[np.argmax(best_test_acc)]
    ytrain_acc = np.amax(best_test_acc)

    xtrain_acc3 = epoch[np.argmax(best_test_acc[624:924])]+624
    ytrain_acc3 = np.amax(best_test_acc[624:924])

    text= "Accuracy={:.2f}".format(ytrain_acc)
    text= "Accuracy={:.2f}".format(ytrain_acc3)
    # fig= plt.figure()

    a=plt.plot( epoch, best_train_acc, color='b', label="Training accuracy")
    b=plt.plot( epoch, best_test_acc, color='c', label="Testing accuracy")
    plt.tick_params(axis='y')
    plt.ylim([65, 98])
    plt.title("Accuracy", fontsize=30 , weight='bold')
    plt.ylabel("Accuracy",fontsize=25)
    plt.xlabel("Epoch",fontsize=25)
    # annot_max(xtrain_acc, ytrain_acc)
    plt.annotate(text, xy=(xtrain_acc, ytrain_acc), xytext=(xtrain_acc+5, ytrain_acc+1),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    plt.annotate(text, xy=(xtrain_acc3, ytrain_acc3), xytext=(xtrain_acc3-2, ytrain_acc3+1),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')

    plt.annotate("Accuracy="+str(best_test_acc[424]), xy=(593, best_test_acc[424]), xytext=(593+20, best_test_acc[424]-2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')
    plt.annotate("Accuracy="+str(best_test_acc[524]), xy=(693, best_test_acc[524]), xytext=(693+20, best_test_acc[524]-2),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')    
    plt.annotate("Accuracy="+str(best_test_acc[624]), xy=(794, best_test_acc[624]), xytext=(794+5, best_test_acc[624]+1),arrowprops=dict(facecolor='black',arrowstyle='->'), fontsize=15, weight='bold')    

    plt.legend(shadow=True, fancybox=True)
    plt.show()
    