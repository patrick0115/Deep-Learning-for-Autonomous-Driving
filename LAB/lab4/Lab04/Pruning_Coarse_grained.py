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
warnings.simplefilter(action='ignore', category=UserWarning)


def train(model, epoch,data_loader,data_set,data_loader_aug,data_set_aug,device,optimizer):
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
  


   
    train_loss = float(total_loss) / (len(train_loader)+len(train_loader_aug))
    train_acc = 100.0 * float(correct) / (len(data_set)+len(data_set_aug))
    # print('Epochhhh: %3d' % epoch, '|train loss: %.4f' % train_loss, '|train accuracy: %.2f' % train_acc)
    return train_acc,train_loss 


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
    




if __name__ == '__main__':
    # Parameter ------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=256,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=300,        help='number of epochs')
    parser.add_argument('-l','--lr',       type=float,            default=0.0001,        help='number of learning rate')
    parser.add_argument('-p','--pruning',       type=float,            default=0.2,        help='number of pruning ')
    parser.add_argument('-n','--name',       type=str,            default='coarse_1_',        help='number data ')
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
    cfg = checkpoint['cfg']
    # print(cfg)
    # calulate parameter  ------------------------------------------------------------------------------------
    print(model_org)
    # count_parameters(model_org)
    total_param = sum([param.nelement() for param in model_org.parameters()])
    print("Number of parameter before pruning: %.2fk" % (total_param/1e3))
    print('Accuracy before pruning')
    test_acc = test(model_org)
    print(round(test_acc,2))       
    print(device)

    cfgs = []         #example format:  [125, 120, 155, 403]
    cfgs_mask = [] 
    # for i, param in enumerate(model_org.parameters()):
    #     print(param)
    BN_torch = torch.tensor([]).to(device)
    for m in model_org.modules():
        if isinstance(m, nn.BatchNorm1d):
            BN_torch=torch.cat((BN_torch,m.weight.data),0)

    sorted_weight = torch.sort(BN_torch)[0]
    thres_index = int(len(sorted_weight) * pruning)
    thres = sorted_weight[thres_index]
    print(thres)

    for m in model_org.modules():
        if isinstance(m, nn.BatchNorm1d):
            cfg = (m.weight.data > thres).sum().to(device)
            mask = (m.weight.data > thres).to(device)
            # cfgs.append(len(cfg))
            cfgs.append(cfg.item())
            cfgs_mask.append(mask)
    
    # print("[128, 128, 256, 512]")
    print(cfgs)
    print('Pre-processing Successful!')
    print(device)



    new_model = M5(cfgs).to(device)
    old_modules = list(model_org.modules())
    new_modules = list(new_model.modules())

    layer_id_in_cfg = 0
    start_mask = torch.ones(1, dtype = bool)
    end_mask = cfgs_mask[layer_id_in_cfg]

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm1d):
            m1.weight.data = m0.weight.data[end_mask].clone()
            m1.bias.data = m0.bias.data[end_mask].clone()
            m1.running_mean = m0.running_mean[end_mask].clone()
            m1.running_var = m0.running_var[end_mask].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfgs_mask): #prevent out of range
                end_mask = cfgs_mask[layer_id_in_cfg]
                
        elif isinstance(m0, nn.Conv1d):
            w1 = m0.weight.data[:, start_mask, :].clone()
            w1 = w1[end_mask, :, :].clone()
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data[end_mask]

        elif isinstance(m0, nn.Linear):
            m1.weight.data = m0.weight.data[:, start_mask].clone()
            m1.bias.data = m0.bias.data.clone()
            

    localtime = time.asctime( time.localtime(time.time()) )
    timecode=localtime[9:10]+"_"+localtime[11:13]+"_"+localtime[14:16]
    
    # save  coarse  model  ------------------------------------------------------------------------------------
    torch.save({'cfg': cfgs, 'state_dict': new_model.state_dict()}, './Checkpoint/'+name+str(pruning)+'_'+str(timecode)+'_batchsize_'+str(batchsize)+'.pth.tar')
   


    total_param = sum([param.nelement() for param in new_model.parameters()])
    print("Number of parameter after pruning: %.2fk" % (total_param/1e3))
    print('Accuracy after pruning, before fine-tuning')
    test_acc = test(new_model)
    print(round(test_acc,2))




    # declare optimizer and loss function ------------------------------------------------------------------------------------
    optimizer = optim.Adam(new_model.parameters(), lr=lr  )
    print('start finetuning')


    # create record file  ------------------------------------------------------------------------------------
    localtime = time.asctime( time.localtime(time.time()) )
    timecode=localtime[9:10]+"_"+localtime[11:13]+"_"+localtime[14:16]
    checkpoint = open('./Checkpoint/'+name+str(timecode)+'_batchsize_'+str(batchsize)+'.txt', 'w')

    # Finetuning  ------------------------------------------------------------------------------------

    best_accuracy = 0

    for epoch in range(1, Epoch + 1):
        train_acc ,train_loss= train(new_model, epoch,train_loader,train_set,train_loader_aug,train_set_aug,device,optimizer)
        test_acc = test(new_model)
        # total_params,sparsity_train= count_parameters(model_fg,show=False)
        
        print('Epoch: %3d' % epoch, '|train loss: %.4f' % train_loss, '|train accuracy: %.2f' % train_acc,'|test_acc:  %.2f' % test_acc)
        
        checkpoint = open('./Checkpoint/'+name+str(timecode)+'_batchsize_'+str(batchsize)+'.txt', 'a')
        checkpoint.write('Epoch: %3d' % epoch + '|train loss: %.4f' % train_loss+ '|train accuracy: %.2f' % train_acc+ '|test accuracy: %.2f' % test_acc+'\n')        
        checkpoint.close()
        if test_acc > best_accuracy:
            
            print('Saving..')
            torch.save({'cfg': new_model.cfg, 'state_dict': new_model.state_dict()}, './Checkpoint/'+name+str(timecode)+'_batchsize_'+str(batchsize)+'.pth.tar')
            best_accuracy = test_acc
        
        # print('Epoch: %3d' % epoch, '|train loss: %.4f' % train_loss, '|train accuracy: %.2f' % train_acc,'|test accuracy: %.2f' % test_acc,'|best accuracy: %.2f' % best_accuracy)  
    print('Best accuracy: %.2f' % best_accuracy)