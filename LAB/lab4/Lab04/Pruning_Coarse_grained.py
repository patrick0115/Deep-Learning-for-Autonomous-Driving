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
import torch.nn.utils.prune as prune
import copy
from tqdm import tqdm
import time 
from prettytable import PrettyTable
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train(model, epoch,data_loader,data_set,data_loader_aug,data_set_aug,device,optimizer,mask_list):
    model.train()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss() 
    print("----------------------------------------------------------------------------------------------------")

    for data, target in tqdm(data_loader):
              
        data = data.to(device)
        target = target.to(device)


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
        a=0  
        # wlist=[]
        length = len(list(model.parameters()))
        for i, param in enumerate(model.parameters()):
            if len(param.size())!=1 and i<length-2:
                # print(param.detach().cpu().numpy())
                weight = param.detach().cpu().numpy()
                weight[mask_list[a]] = 0       
                weight = torch.from_numpy(weight).to(device)
                param.data = weight
                a=a+1

    for data, target in tqdm(data_loader_aug):
              
        data = data.to(device)
        target = target.to(device)


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
        a=0  
        # wlist=[]
        length = len(list(model.parameters()))
        for i, param in enumerate(model.parameters()):
            if len(param.size())!=1 and i<length-2:
                # print(param.detach().cpu().numpy())
                weight = param.detach().cpu().numpy()
                weight[mask_list[a]] = 0       
                weight = torch.from_numpy(weight).to(device)
                param.data = weight
                a=a+1


    _,sparsity_train= count_parameters(model,show=False)
    train_loss = float(total_loss) / (len(train_loader)+len(train_loader_aug))
    train_acc = 100.0 * float(correct) / (len(data_set)+len(data_set_aug))
    # print('Epochhhh: %3d' % epoch, '|train loss: %.4f' % train_loss, '|train accuracy: %.2f' % train_acc)
    return train_acc,train_loss ,sparsity_train


def test(model):
    model.eval()
    correct = 0

    for data, target in test_loader:

        data = data.to(device)
        target = target.to(device)

        #forward
        output = model(data)

        pred = output.argmax(dim=-1)
        correct += pred.squeeze().eq(target).sum().item()

    # print testing stats
    test_acc = 100.0 * float(correct) / len(test_set)
    # print('test accuracy: %.2f' % test_acc)
    return test_acc

def count_parameters(model,show=True):
    table = PrettyTable(["Modules", "Parameters","zero number","Sparsity"])
    total_params = 0
    org_w=[]
    org_z=[]
    length = len(list(model.parameters()))
    for i, (name, parameter )in enumerate(model.named_parameters()):
   
        if not parameter.requires_grad: continue
        w = parameter.detach().cpu().numpy()    
        z_num=np.sum(np.sum(np.where(w,0,1)))
        if len(parameter.size())!=1 and i<length-2:
            org_w.append(parameter.numel())
            org_z.append(z_num)
        params = parameter.numel()
        table.add_row([name, params,z_num,str(round((z_num/params)*100,1))+" %"])
        total_params+=params
    sparsity=round(np.sum(org_z)/np.sum(org_w),1)
    if show:
        print(table)
        print(f"Total Trainable Params: {total_params}, Sparsity: {sparsity*100} %")

    return total_params,sparsity
    
def Pruning_fg(model,p_pruning=60):
    model_fg = copy.deepcopy(model)
    length = len(list(model_fg.parameters()))
    mask_list=[]
    for i, param in enumerate(model_fg.parameters()):
        if len(param.size())!=1 and i<length-2:
            weight = param.detach().cpu().numpy()
            w_mask=np.abs(weight)<np.percentile(np.abs(weight),p_pruning)
            mask_list.append(w_mask)
            weight[w_mask] = 0       
            weight = torch.from_numpy(weight).to(device)
            param.data = weight
    return model_fg,mask_list

def pruning(new_tensor1):    
    #sort the weight matrix in each layer
    sorted_tensor1= np.sort(new_tensor1[new_tensor1!=0],axis=None)
    # 0.2 means the pruning rate in this layer
    cutoff_index1 = np.round(0.2 * sorted_tensor1.size).astype(int)
    cutoff1 = sorted_tensor1[cutoff_index1]
    new_tensor1=np.where(new_tensor1 < cutoff1, 0, new_tensor1)
    
    
    return new_tensor1


if __name__ == '__main__':
    # Parameter ------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=256,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=100,        help='number of epochs')
    parser.add_argument('-l','--lr',       type=float,            default=0.00001,        help='number of learning rate')
    parser.add_argument('-p','--pruning',       type=float,            default=60,        help='number of pruning ')
    parser.add_argument('-n','--name',       type=str,            default='fine_grained_1_',        help='number data ')
    opt = parser.parse_args()

    batchsize   = opt.batchsize
    Epoch       = opt.epoch
    lr       = opt.lr
    pruning       = opt.pruning
    name       = opt.name

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)


    # declare dataloader  ------------------------------------------------------------------------------------
    training_params = {"batch_size":batchsize ,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": 1}

    testing_params = {"batch_size": batchsize,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}

    train_set = SpeechCommandDataset()
    train_loader = DataLoader(train_set, **training_params) 

    train_set_aug = SpeechCommandDataset(aug=True)
    train_loader_aug = DataLoader(train_set_aug, **training_params) 

    test_set = SpeechCommandDataset(is_training=False)
    test_loader = DataLoader(test_set, **testing_params)

    # load model  ------------------------------------------------------------------------------------
    model_path = './log/best_model_clean.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location = device)
    model_org = M5(cfg = checkpoint['cfg']).to(device)
    model_org.load_state_dict(checkpoint['state_dict'])

    # calulate parameter  ------------------------------------------------------------------------------------
    print(model_org)
    count_parameters(model_org)

    print('\nAccuracy before pruning')
    test_acc = test(model_org)
    print(round(test_acc,2))
    
    # for m in model_org.modules():
    #     # print(m)
    #     if isinstance(m, nn.BatchNorm1d):
    #         # print(m)
    #         print(m.weight.data)
    cfgs = []         #example format:  [125, 120, 155, 403]
    cfgs_mask = [] 
    # sorted_weight = torch.sort()[0]
    # # thres_index = int(sorted_weight * pruning)
    # # thres = sorted_weight[thres_index]
    # # print(thres)
    print("------------------------------------")
    for m in model_org.modules():
        if isinstance(m, nn.BatchNorm1d):
            cfg = (m.weight.data > 1).sum()
            mask = m.weight.data > 1
            cfgs.append(cfg)
            cfgs_mask.append(mask)
    print(cfgs)
    print(cfgs_mask)

