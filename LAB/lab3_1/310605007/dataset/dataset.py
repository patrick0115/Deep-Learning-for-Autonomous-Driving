import numpy as np
import os
import importlib
import scipy.misc as sm
import torch
from torch.utils.data import Dataset

class CityScapesDataset(Dataset):
        
    def __init__(self,root,split,img_size,task,augmentation=None):
        self.root         = root
        self.split        = split
        self.img_size     = img_size
        self.task         = task
        self.augmentation = augmentation
        self.path         = os.path.join(self.root,self.split)
        self.filename     = []
        if self.task == 'all':
            self.LABELS = importlib.import_module('utils.labels_all')
        elif self.task == 'cat':
            self.LABELS = importlib.import_module('utils.labels_cat')
        elif self.task == 'road':
            self.LABELS = importlib.import_module('utils.labels_road')

        all_filename = os.listdir(self.image_dir)
        for i in range(len(all_filename)):
            self.filename.append('%04d.png'%(i))
        
    @property
    def image_dir(self):
        return os.path.join(self.path,'image')

    @property
    def label_dir(self):
        return os.path.join(self.path,'semantic_rgb')

    def encode_label(self,label):
        mask = np.zeros(label.shape[0:2])
        for _label in self.LABELS.labels:
            mask[np.sum(label == np.array(_label.color),axis=2)==3] = _label.trainId  
        return mask

    def __getitem__(self,index):
        img = sm.imread(os.path.join(self.image_dir,self.filename[index]),mode='RGB')
        img = sm.imresize(img,self.img_size,'nearest')
        img = np.array(img,dtype=np.uint8)
        if self.split is not 'detect':
            label = sm.imread(os.path.join(self.label_dir,self.filename[index]),mode='RGB')
        #'''
        if self.split is  'training' :
            label = sm.imresize(label,self.img_size,'nearest')
        if self.split is  'validation' : 
            label = sm.imresize(label,self.img_size,'nearest')
        #'''
        if self.split is not 'detect':
            label = np.array(label,dtype=np.uint8)
            label = self.encode_label(label)
        if self.augmentation != None:
            img, label = self.augmentation(img,label)
        img = np.transpose(img,(2,0,1))
        if self.split is not 'detect':
            # label = sm.imresize(label,self.img_size,'nearest')
            return torch.from_numpy(img).float()/255,torch.from_numpy(label).long() 
        else:
            return torch.from_numpy(img).float()/255
    def __len__(self):
        return len(self.filename)