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
def test(model,dataloader,test_set):
    correct = 0
    with torch.no_grad():
        model.eval()
        # bs = dataloader_test.batch_size
        result = []
        # targets=[]
        for i, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1, keepdim=True)
            # targets.append(target)
            arr = preds.data.cpu().numpy()
            correct += preds.squeeze().eq(target).sum().item()
            # print(preds)
            for j in range(preds.size()[0]):
                # file_name = dataset_test.samples[i*bs+j][0].split('/')[-1]?
                result.append(preds[j].cpu().numpy()[0])
        # print(result)
            # results.extend(result) 
        test_acc = 100.0 * float(correct) / len(test_set)
    return result,test_acc

def plot_confusion_matrix(y_pred,y_true ):
    plt.figure()
    cnf_matrix = confusion_matrix(y_true, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    plt.imshow(cnf_matrix, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, list(range(10)) , rotation=45)
    plt.yticks(tick_marks, list(range(10)) )
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    return cnf_matrix

def test_eval(model_path):
    ## all 23701  test 4735 train 18947
    testing_params = {"batch_size": batchsize,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}
    test_set = SpeechCommandDataset(is_training=False)
    test_loader = DataLoader(test_set, **testing_params)   

    # model_path = './Checkpoint/best_model_org.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location = device)
    model = M5(cfg = checkpoint['cfg']).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    y_pred_test=[]
    y_true_test=[]

    for i in range(len(test_set)):
        y_true_test.append(int(test_set[i][1]))
    y_pred_test,acc= test(model,test_loader,test_set)

    return y_pred_test ,y_true_test,acc

def train_eval(model_path):
    ## all 23701  test 4735 train 18947
    training_params = {"batch_size":batchsize ,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}

    train_set = SpeechCommandDataset()
    train_loader = DataLoader(train_set, **training_params)  

    # model_path = './Checkpoint/best_model_org.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location = device)
    model = M5(cfg = checkpoint['cfg']).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    y_pred_train=[]
    y_true_train=[]
   

    for i in range(len(train_set)):
        y_true_train.append(int(train_set[i][1]))

    y_pred_train,acc= test(model,train_loader,train_set)
    return y_pred_train ,y_true_train,acc



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=256,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=10000,        help='number of epochs')
    parser.add_argument('-l','--lr',       type=float,            default=0.0001,        help='number of learning rate')
    opt = parser.parse_args()

    batchsize   = opt.batchsize
    Epoch       = opt.epoch
    lr       = opt.lr

    model_path = './log/best_model_org.pth.tar'
    y_pred_test ,y_true_test,acc =test_eval(model_path)
    print(acc)
    a=plot_confusion_matrix(y_pred_test,y_true_test)


    # # test_data=[]
    # # train_data=[]
    # # f_test = open('./speech_commands/test_list.txt')
    # # f_train = open('./speech_commands/train_list.txt')
    # # for line_test in f_test.readlines():
    # #     test_data.append(line_test)
    # # for line_train in f_train.readlines():
    # #     train_data.append(line_train)
    # # f_test.close
    # # f_train.close
    # # print(len(test_data))
    # # print(len(train_data))

    # # print(len(train_set))
    # # print(len(train_loader))








    # # train_list_txt = open('./speech_commands/train_list.txt', 'r')

 
    # # train_list_txt.close()




    # print("---")
    # # print(y_pred_train)
    # print("--55555-")

    # train_list_txt_wrong = open('./speech_commands/train_list_wrong.txt', 'r')
    # for line in train_list_txt_wrong.read().splitlines():        
    #     a=os.path.join('./speech_commands/',line) 
    #     shutil.copyfile(a , './speech_commands_wrong_train/'+line)

    # for i in range(len(y_pred_train)):  
    #     if y_pred_train[i]!=y_true_train[i]:
    #         # print(train_set[i][2])
    #         train_list_txt_wrong.write(train_set[i][2]+'\n')

    # train_list_txt_wrong.close()





    # # print(y_pred_train)
    # test_wrong_source=[]
    # test_wrong_destination=[]
    # for line in test_txt.read().splitlines():
    #     path_source=os.path.join('./speech_commands/',line) 
    #     path_destination=os.path.join('./speech_commands_wrong/',line) 
    #     test_wrong_source.append(path_source) 
    #     test_wrong_destination.append(path_destination) 
    # test_txt.close()
    # data_list_wrong_path="./speech_commands/train_list_wrong.txt"
    # data_list_path="./speech_commands/train_list.txt"
    # ids = [id.strip() for id in open(data_list_path)]
    # ids_wrong = [id_wrong.strip() for id_wrong in open(data_list_wrong_path)]
    # # print(ids)
    # # print(ids_wrong)
    # print(len(ids))
    # print(len(ids_wrong))
    # a=set(ids).difference(ids_wrong)
    # # print(a)
    # print(len(a))
    # train_list_txt_new = open('./speech_commands/train_list_new.txt', 'w')
    
    # for c,b in enumerate(a):
    #     print(b)
    #     train_list_txt_new.write(b+"\n")        
    # test_list_txt_new = open('./speech_commands/test_list_new.txt', 'w')
    # train_txt.close()
    # print(train_wrong)

    # os.mkdir('./speech_commands_wrong_train/down')
    # os.mkdir('./speech_commands_wrong_train/go')
    # os.mkdir('./speech_commands_wrong_train/left')
    # os.mkdir('./speech_commands_wrong_train/no')
    # os.mkdir('./speech_commands_wrong_train/off')
    # os.mkdir('./speech_commands_wrong_train/on')
    # os.mkdir('./speech_commands_wrong_train/right')
    # os.mkdir('./speech_commands_wrong_train/stop')
    # os.mkdir('./speech_commands_wrong_train/up')
    # os.mkdir('./speech_commands_wrong_train/yes')


    # wrong_data=[]
    # train_list_txt = open('./speech_commands/train_list.txt', 'r')
    # test_list_txt = open('./speech_commands/test_list.txt', 'r')
    # wrong_txt = open('./speech_commands/wrong_all.txt', 'r')
    # train_list_txt_np = []
    # test_list_txt_np = []
    # wrong_txt_np = []

    # for line in train_list_txt.read().splitlines():
    #     train_list_txt_np.append(line)
    # train_list_txt.close()

    # for line1 in test_list_txt.read().splitlines():
    #     test_list_txt_np.append(line1)
    # test_list_txt.close()

    # for line2 in wrong_txt.read().splitlines():
    #     wrong_txt_np.append(line2)
    # wrong_txt.close()

    # print(len(train_list_txt_np))
    # print(len(test_list_txt_np))
    # print(len(wrong_txt_np))

    # for i in range(len(test_list_txt_np)):
    #     for j in range(len(wrong_txt_np)):
    #         #
    #         # print(train_list_txt_np[i],wrong_txt_np[j])
    #         if test_list_txt_np[i]==wrong_txt_np[j]:
    #             print(test_list_txt_np[i])


    # yes = listdir('./speech_commands_wrong_train/yes')
    # no = listdir('./speech_commands_wrong_train/no')
    # up = listdir('./speech_commands_wrong_train/up')
    # down = listdir('./speech_commands_wrong_train/down')
    # left = listdir('./speech_commands_wrong_train/left')
    # right = listdir('./speech_commands_wrong_train/right')
    # on = listdir('./speech_commands_wrong_train/on')
    # off = listdir('./speech_commands_wrong_train/off')
    # stop = listdir('./speech_commands_wrong_train/stop')
    # go = listdir('./speech_commands_wrong_train/go')
    # train_list_txt = open('./speech_commands_wrong_train/train_list_wrong.txt', 'w')
    
    # # # print(type(go))
    # list_data_dir=[yes,no,up,down,left,right,on,off,stop,go]
    # list_data_dir_name=["yes","no","up","down","left","right","on","off","stop","go"]
    # for i,a in enumerate(list_data_dir):
    #     for f in a:
    #         name=list_data_dir_name[i]+"/"+f
    #         print(name)
    #         train_list_txt.write(name+'\n')
            # # print(f)

    # # for f in yes:
    # #     fullpath = join('./speech_commands_wrong/yes', f)
    # #     if isfile(fullpath):
    # #         wrong_data.append('yes/'+f+'\n')


            
            
            
    # for i in range(len(wrong_data)):
    #     wrong_txt.write(wrong_data[i])
    # wrong_txt.close()




    # print(len(y_pred_train))
