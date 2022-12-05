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

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # a=ax1.plot( epoch, best_train_acc, color='b', label="Training accuracy")
    # b=ax1.plot( epoch, best_test_acc, color='c', label="Testing accuracy")
    # ax1.tick_params(axis='y')
    # ax1.set_ylim([80, 98])
    # ax1.set_title("Accuracy of best model without dirty data")
    # c=ax2.plot( epoch, best_loss, color='g',label="Loss")
    # ax2.tick_params(axis='y')
    # ax2.set_ylim([0, 0.6])
    # ax1.set_ylabel("Accuracy")
    # ax2.set_ylabel("Loss")
    # ax1.set_xlabel("Epoch")
    # ax1.legend(shadow=True, fancybox=True)
    # ax2.legend(shadow=True, fancybox=True)
    # plt.show()
    


    # fine_train_acc=[]
    # fine_loss=[]
    # fine_test_acc=[]
    # epoch=[]
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

    xtrain_acc = epoch[np.argmax(best_test_acc)]
    ytrain_acc = np.amax(best_test_acc)
    print(xtrain_acc)
    print(ytrain_acc)
    text= "Accuracy={:.2f}".format(ytrain_acc)

    fig= plt.figure()

    a=plt.plot( epoch, best_train_acc, color='b', label="Training accuracy")
    b=plt.plot( epoch, best_test_acc, color='c', label="Testing accuracy")
    plt.tick_params(axis='y')
    plt.ylim([70, 98])
    plt.title("Accuracy of best model without dirty data")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    # annot_max(xtrain_acc, ytrain_acc)
    plt.annotate(text, xy=(xtrain_acc, ytrain_acc), xytext=(xtrain_acc+20, ytrain_acc+0.5),arrowprops=dict(facecolor='black',arrowstyle='->'))
    # plt.annotate(text, xy=(xtrain_acc, ytrain_acc), xytext=(0.94,0.96))
    plt.legend(shadow=True, fancybox=True)
    plt.show()
    