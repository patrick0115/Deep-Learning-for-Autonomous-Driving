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

from sklearn.metrics import confusion_matrix
import itertools
def test(model,dataloader_test):
    with torch.no_grad():
        model.eval()
        bs = dataloader_test.batch_size
        result = []
        # targets=[]
        for i, (data, target) in enumerate(dataloader_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, preds = torch.max(output, 1, keepdim=True)
            # targets.append(target)
            arr = preds.data.cpu().numpy()
            for j in range(preds.size()[0]):
                # file_name = dataset_test.samples[i*bs+j][0].split('/')[-1]?
                result.append(preds[j].cpu().numpy()[0])
    return result

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
    testing_params = {"batch_size": batchsize,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}
    training_params = {"batch_size":batchsize ,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}
    test_set = SpeechCommandDataset(is_training=False)
    test_loader = DataLoader(test_set, **testing_params)
    train_set = SpeechCommandDataset()
    train_loader = DataLoader(train_set, **training_params) 

    # test_model =  np.load("./archive/data.pkl",allow_pickle=True).to(device)
    model = M5().to(device)
    model.load_state_dict(torch.load("./Checkpoint/best_model.pth.tar")['state_dict'])
    
    y_pred_test=[]
    y_true_test=[]
    y_pred_train=[]
    y_true_train=[]

    for i in range(len(test_set)):
        y_true_test.append(int(test_set[i][1]))
    for i in range(len(train_set)):    
        y_true_train.append(int(train_set[i][1]))
        # print(train_set[i][1])

    y_pred_test= test(model,test_loader)
    y_pred_train= test(model,train_loader)
    # print(y_pred_train)
    # print("----------------------------")
    # print(y_true_train)   
    a=plot_confusion_matrix(y_pred_test,y_true_test)
    b=plot_confusion_matrix(y_pred_train,y_true_train)
    print(a)
    print(b)