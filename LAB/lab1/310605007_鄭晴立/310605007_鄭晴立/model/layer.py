import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.uniform(-1, 1, (in_features, out_features)) * 0.01
        self.bias = np.random.uniform(-1, 1, (1, out_features))
        print(self.weight )
        self.in_features = in_features
        self.out_features = out_features
        # self.trained = 0
        # self.weight_grad=None
        # self.bias_grad=None
    def forward(self, input):
        self.inputs = input
        self.batch_size = input.shape[0]
        # print(input.shape)
        # print(self.weight.shape)
        # print(self.bias.shape)
        return np.dot(input, self.weight) + self.bias



    def backward(self, output_grad):
        # dx = np.dot(output_grad, np.transpose(self.weight))        
        # # lr = opti.lr
        # dw = np.dot(np.transpose(self.inputs), output_grad) / self.batch_size
        # db = np.sum(output_grad, axis=0) / self.batch_size   
        input_grad =  np.dot(output_grad, np.transpose(self.weight))
        # print(output_grad)
        self.weight_grad = np.dot(np.transpose(self.inputs), output_grad) / self.batch_size
        self.bias_grad = np.sum(output_grad, axis=0) / self.batch_size
        # print( self.weight_grad )
        return input_grad

## by yourself .Finish your own NN framework
class ACTIVITY1(_Layer):
    def __init__(self):
        pass
    def forward(self, input:np.array):
        self.inputs = input
        return self.relu(input)    
        # return output
    def relu(self, x:np.array):
        return np.maximum(x, 0)
    def backward(self, dy:np.array):
        dx = (self.inputs > 0) * dy
        return dx      

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass
    # sofmax activation
    def softmax(self,X):
        exps = np.exp(X - np.max(X,axis=1).reshape(-1,1))
        return exps / np.sum(exps,axis=1)[:,None]

    # derivative of softmax
    def softmax_derivative(self,pred):
        return pred * (1 -(1 * pred).sum(axis=1)[:,None])
    def cross_entropy(self,X,y):
        X = X.clip(min=1e-8,max=None)
        # print('\n\nCE: ', (np.where(y==1,-np.log(X), 0)).sum(axis=1))
        return (np.where(y==1,-np.log(X), 0)).sum(axis=1)

    def cross_entropy_derivative(self,X,y):
        X = X.clip(min=1e-8,max=None)
        # print('\n\nCED: ', np.where(y==1,-1/X, 0))
        return np.where(y==1,-1/X, 0)
    def forward(self, input:np.array, target):
        # print(input.shape)
        self.inputs = input
        predict = self.softmax(input)     
        your_loss = self.cross_entropy(predict,target)
        # print(your_loss)
        '''Average loss'''

        return predict, your_loss 
    def backward(self,output_grad,target):
        dloss = self.cross_entropy_derivative(output_grad, target) 
        input_grad = dloss*self.softmax_derivative(output_grad)
        return input_grad
    
    