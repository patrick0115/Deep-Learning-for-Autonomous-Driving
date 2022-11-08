# from sklearn.svm import l1_min_c
from .layer import *

class optimizer():
    def __init__(self, name = 'SGD', lr = 0.001, hyper_parameter = {}):
        self.name = name
        self.lr = lr
        self.hyper_parameter = hyper_parameter

class Network(object):
    def __init__(self,layers,Learning_rate,opt):
        self.opt=opt
        self.lr = Learning_rate
        self.layers = layers
        self.layerW =[]
        self.layerB =[]
        # print(layers[0].weight )
        for i in range(0,len(layers),2):
            self.layerW.append(layers[i].weight)
        for i in range(0,len(layers),2):
            self.layerB.append(layers[i].bias)
            
    def forward(self, input, target):
        self.f=[]
        self.act_f=[]
        self.pred = input
        for i in range(len(self.layers)):
            self.pred = self.layers[i].forward(self.pred)
            if i % 2 == 1:
                self.f.append(self.pred)
            else:
                self.act_f.append(self.pred)
        self.f.reverse()
        self.act_f.reverse()
        self.loss_value=self.layers[-1].loss(target)      

        return self.pred, self.loss_value

    def backward(self, target):

        self.h_last_grad = self.layers[-1].backward(self.pred, target)
        grad=self.h_last_grad 
        for i in range(int((len(self.layers))/2)-1):
            act_grad=self.layers[-3-2*i].backward(self.act_f[i+1])
            grad=self.layers[-2-2*i].backward(act_grad,grad,self.lr,self.opt)

