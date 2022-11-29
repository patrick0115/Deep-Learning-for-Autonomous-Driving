import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from speech_command_dataset import SpeechCommandDataset
import numpy as np
import matplotlib.pyplot as plt
from model import M5
import argparse
from tqdm import tqdm
# from IPython.display import Audio
# import torchaudio
# import torchaudio.functional as F
from augmentation import aug
def train(model, epoch,data_loader,device):
    model.train()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss() 
    
    for data, target in tqdm(data_loader):
        
        data = data.to(device)
        target = target.to(device)
        # data=aug(data)
        #forward
        output = model(data)

        target = target.to(torch.int64) 
        loss = criterion(output, target)
        
        total_loss += loss.item()
        pred = output.argmax(dim=-1)
        correct += pred.squeeze().eq(target).sum().item()
        
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print training stats
    train_loss = float(total_loss) / len(train_loader)
    train_acc = 100.0 * float(correct) / len(train_set)
    print('Epoch: %3d' % epoch, '|train loss: %.4f' % train_loss, '|train accuracy: %.2f' % train_acc)
    return train_acc


def test(model, epoch,data_loader,device):
    model.eval()
    correct = 0
    for data, target in tqdm(data_loader):

        data = data.to(device)
        target = target.to(device)

        #forward
        output = model(data)

        pred = output.argmax(dim=-1)
        correct += pred.squeeze().eq(target).sum().item()
        
    # print testing stats
    test_acc = 100.0 * float(correct) / len(test_set)
    print('Epoch: %3d' % epoch, '|test accuracy: %.2f' % test_acc)
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=256,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=10000,        help='number of epochs')
    parser.add_argument('-l','--lr',       type=float,            default=0.00005,        help='number of learning rate')
    opt = parser.parse_args()

    batchsize   = opt.batchsize
    Epoch       = opt.epoch
    lr       = opt.lr

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)


    # declare dataloader
    training_params = {"batch_size":batchsize ,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": 1}

    testing_params = {"batch_size": batchsize,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}



    test_set = SpeechCommandDataset(is_training=False)
    test_loader = DataLoader(test_set, **testing_params)




    # declare network
    model = M5().to(device)
    # print(model)

    # declare optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr )

    print('start training')

    best_accuracy = 0
        
    for epoch in tqdm(range(1, Epoch + 1)):
            
        train_set = SpeechCommandDataset()
        train_loader = DataLoader(train_set, **training_params)
        train_acc = train(model, epoch,train_loader,device)
        test_acc = test(model, epoch,test_loader,device)
        
        if test_acc > best_accuracy:
            print('Saving..')
            torch.save({'cfg': model.cfg, 'state_dict': model.state_dict()}, './Checkpoint/best_model.pth.tar')
            best_accuracy = test_acc
            
    print('Best accuracy: %.2f' % best_accuracy)