import os
import time
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo4 import YoloBody
from nets.yolo_training import Generator, YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1,3,2])[::-1,:,:]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,genval,Epoch,cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]


            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            num_pos_all = 0

            for i in range(3):
                loss_item, num_pos = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    net.eval()
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                num_pos_all = 0
                for i in range(3):
                    loss_item, num_pos = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos
                loss = sum(losses) / num_pos_all
                val_loss += loss.item()
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batchsize',   type=int,            default=8,          help='input batch size')
    parser.add_argument('-e','--epoch',       type=int,            default=80,        help='number of epochs')
    parser.add_argument('-i','--img-size',    nargs='+', type=int, default=[256, 512], help='resize to imgsize') # [256, 512]
    parser.add_argument('-m','--model-name',  type=str,            default='model',    help='for name of save model')
    parser.add_argument('-o','--output-path', type=str,            default='log',      help='output directory(including log and savemodel)')
    parser.add_argument('-r','--resume',      type=str,            default=None,       help='the file name of checkpoint you want to resume')
    parser.add_argument('-t','--task',        type=str,            default='cat',      help='the training task: cat')
    parser.add_argument('-l','--learningrate',type=float,          default=0.001,       help='input learningrate')
    parser.add_argument('-lo','--lossfunction',   type=int,            default=0,          help='0=CrossEntropyLoss 1=dice_loss')
    parser.add_argument('-mo','--modeltype',   type=int,            default=0,          help='0=U-Net 1=U-Net++')
    opt = parser.parse_args()
    batchsize   = opt.batchsize
    Epoch       = opt.epoch
    img_size    = tuple(opt.img_size)
    model_name  = opt.model_name
    output_path = opt.output_path
    resume      = opt.resume
    task        = opt.task
    lr          = opt.learningrate
    loss_fn     = opt.lossfunction
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modeltype   = opt.modeltype



    backbone = "mobilenetv1"

    pretrained = False
    Cuda = True
    Use_Data_Loader = True

    normalize = True
    #-------------------------------#
    #   416x416 or   608x608
    #-------------------------------#
    input_shape = (416,416)

    #----------------------------------------------------#
    #   classes???anchor????????????????????????
    #   ????????????????????????classes_path?????????????????????????????????
    #----------------------------------------------------#
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/car_classes.txt'   
    #----------------------------------------------------#
    #   ??????classes???anchor
    #----------------------------------------------------#
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    
    #------------------------------------------------------#
    #   Yolov4???tricks??????
    #   mosaic ????????????????????? True or False
    #   Cosine_scheduler ????????????????????? True or False
    #   label_smoothing ???????????? 0.01???????????? ???0.01???0.005
    #------------------------------------------------------#
    mosaic = True
    Cosine_lr = True
    smoooth_label = 0.005

    #------------------------------------------------------#
    #   ??????yolo??????
    #   ????????????????????????classes_path????????????txt??????
    #------------------------------------------------------#
    model = YoloBody(len(anchors[0]), num_classes, backbone=backbone, pretrained=pretrained)

    #------------------------------------------------------#
    #   ??????????????????README?????????????????????
    #------------------------------------------------------#
    model_path = "model_data/yolov4_mobilenet_v1_map76.62.pth"
    # ???????????????????????????
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # ??????loss??????
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda, normalize))

    #----------------------------------------------------#
    #   ???????????????????????????
    #----------------------------------------------------#
    annotation_path = 'car_train.txt'
    #----------------------------------------------------------------------#
    #   ?????????????????????train.py??????????????????
    #   2007_test.txt???2007_val.txt?????????????????????????????????????????????????????????
    #   ?????????????????????????????????????????????????????????1:9
    #----------------------------------------------------------------------#
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    #------------------------------------------------------#
    #   ???????????????????????????????????????????????????????????????????????????
    #   ????????????????????????????????????????????????
    #   Init_Epoch???????????????
    #   Freeze_Epoch????????????????????????
    #   Epoch???????????????
    #   ??????OOM???????????????????????????Batch_size
    #------------------------------------------------------#
    if True:
        lr = 1e-3
        # Batch_size = 8
        Batch_size = batchsize
        Init_Epoch = 0
        # Freeze_Epoch = 50
        Freeze_Epoch=Epoch//2
        
        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(train=False, mosaic = mosaic)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   ????????????????????????
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Freeze_Epoch,Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        # Batch_size = 8
        Batch_size = batchsize
        # Freeze_Epoch = 50
        # Unfreeze_Epoch = 100
        Freeze_Epoch = Epoch/2
        Unfreeze_Epoch = Epoch


        optimizer = optim.Adam(net.parameters(),lr,weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (input_shape[0], input_shape[1]), mosaic=mosaic, is_train=True)
            val_dataset = YoloDataset(lines[num_train:], (input_shape[0], input_shape[1]), mosaic=False, is_train=False)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4,pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (input_shape[0], input_shape[1])).generate(train=True, mosaic = mosaic)
            gen_val = Generator(Batch_size, lines[num_train:],
                            (input_shape[0], input_shape[1])).generate(train=False, mosaic = mosaic)

        epoch_size = max(1, num_train//Batch_size)
        epoch_size_val = num_val//Batch_size
        #------------------------------------#
        #   ???????????????
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(net,yolo_losses,epoch,epoch_size,epoch_size_val,gen,gen_val,Unfreeze_Epoch,Cuda)
            lr_scheduler.step()
