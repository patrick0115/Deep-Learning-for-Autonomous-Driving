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

def train(model, epoch,train_loader,train_set,device,optimizer):
    model.train()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss() 
    
    for batch_idx, (data, target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        #forward
        output = model(data)
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

def test(model, epoch,test_loader,test_set,device):
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
    print('Epoch: %3d' % epoch, '|test accuracy: %.2f' % test_acc)
    return test_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=256,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=10000,        help='number of epochs')
    parser.add_argument('-l','--lr',       type=float,            default=0.0001,        help='number of learning rate')
    opt = parser.parse_args()

    batchsize   = opt.batchsize
    Epoch       = opt.epoch
    lr       = opt.lr

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(device)
    training_params = {"batch_size":batchsize ,
                        "shuffle": True,
                        "drop_last": False,
                        "num_workers": 1}

    testing_params = {"batch_size":batchsize ,
                        "shuffle": False,
                        "drop_last": False,
                        "num_workers": 1}

    train_set = SpeechCommandDataset()
    train_loader = DataLoader(train_set, **training_params)

    test_set = SpeechCommandDataset(is_training=False)
    test_loader = DataLoader(test_set, **testing_params)
    model_path = './Checkpoint/best_model.pth.tar'
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location = device)

    model = M5(cfg = checkpoint['cfg']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    total_param = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter before pruning: %.2fk" % (total_param/1e3))

    print('\nAccuracy before pruning')
    test_acc = test(model, 0,test_loader,test_set,device)

    # you can choose your pruning rate
    pruning_rate = 0.1
    #calculate pruning threshold
    #hint
    # model.modules() : go through all layer
    # for m in model.modules():
    #     # print(m)
    #     if isinstance(m, nn.BatchNorm1d):
    #         print(m.weight.data)
    
    lengths = [17,17,19,23,7]
    lengths = torch.Tensor(lengths)
    sorted_weight = torch.sort(lengths)[0]    
  
    print(lengths)
    print(sorted_weight)

    # print(sorted_weight)
    thres_index = int(num_weight * pruning_rate)
    thres = sorted_weight[thres_index]
    print(thres)

    # #get configuration and mask for pruned network
    # #hint
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm1d):
    #         cfg = (m.weight.data > thres).sum()
    #         mask = m.weight.data > thres

    # #original configuration: [128, 128, 256, 512]
    # cfg = []         #example format:  [125, 120, 155, 403]
    # cfg_mask = []    #example format:  [[True, False, ..., True], [False, False, ..., True], [True, False, ..., False], [False, False, ..., True]]


    # print('Pre-processing Successful!')

    # new_model = M5(cfg).to(device)

    # old_modules = list(model.modules())
    # new_modules = list(new_model.modules())

    # layer_id_in_cfg = 0
    # start_mask = torch.ones(1, dtype = bool)
    # end_mask = cfg_mask[layer_id_in_cfg]

    # for layer_id in range(len(old_modules)):
    #     m0 = old_modules[layer_id]
    #     m1 = new_modules[layer_id]
    #     if isinstance(m0, nn.BatchNorm1d):
    #         m1.weight.data = m0.weight.data[end_mask].clone()
    #         m1.bias.data = m0.bias.data[end_mask].clone()
    #         m1.running_mean = m0.running_mean[end_mask].clone()
    #         m1.running_var = m0.running_var[end_mask].clone()
    #         layer_id_in_cfg += 1
    #         start_mask = end_mask.clone()
    #         if layer_id_in_cfg < len(cfg_mask): #prevent out of range
    #             end_mask = cfg_mask[layer_id_in_cfg]
                
    #     elif isinstance(m0, nn.Conv1d):
    #         w1 = m0.weight.data[:, start_mask, :].clone()
    #         w1 = w1[end_mask, :, :].clone()
    #         m1.weight.data = w1.clone()
    #         m1.bias.data = m0.bias.data[end_mask]

    #     elif isinstance(m0, nn.Linear):
    #         m1.weight.data = m0.weight.data[:, start_mask].clone()
    #         m1.bias.data = m0.bias.data.clone()
            
            

    # print('cfg', cfg)
    # torch.save({'cfg': cfg, 'state_dict': new_model.state_dict()}, './Checkpoint/coarse_model.pth.tar')


    # checkpoint = torch.load('./Checkpoint/coarse_model.pth.tar')
    # prune_model = M5(cfg=checkpoint['cfg']).to(device)
    # prune_model.load_state_dict(checkpoint['state_dict'])

    # print(prune_model)

    # total_param = sum([param.nelement() for param in prune_model.parameters()])
    # print("Number of parameter after pruning: %.2fk" % (total_param/1e3))

    # print('\nAccuracy after pruning, before fine-tuning')
    # test_acc = test(prune_model, 0)

    # EPOCH = 
    # LR = 

    # # declare optimizer and loss function
    # optimizer = 
    # print('start finetuning')

    # best_accuracy = 0
        
    # for epoch in range(1, EPOCH + 1):
    #     train_acc = train(prune_model, epoch)
    #     test_acc = test(prune_model, epoch)
        
    #     if test_acc > best_accuracy:
    #         print('Saving..')
    #         torch.save({'cfg': prune_model.cfg, 'state_dict': prune_model.state_dict()}, './Checkpoint/finetuned_model.pth.tar')
    #         best_accuracy = test_acc
            
    # print('Best accuracy: %.2f' % best_accuracy)

    # checkpoint = torch.load('./Checkpoint/finetuned_model.pth.tar')
    # finetuned_model = M5(cfg=checkpoint['cfg']).to(device)
    # finetuned_model.load_state_dict(checkpoint['state_dict'])

    # total_param = sum([param.nelement() for param in finetuned_model.parameters()])
    # print("Number of parameter after pruning: %.2fk" % (total_param/1e3))

    # print('Accuracy after fine-tuning')
    # test_acc = test(finetuned_model, 0)