import numpy as np

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
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.trained = 0  

    def forward(self, input):
        self.input = input
        self.output = np.dot(input,self.weight)+self.bias
        return self.output

    def backward(self, act_grad3,h4_grad,lr,opt):

        self.d = np.multiply(act_grad3, self.weight.dot(h4_grad.transpose()).transpose())

        if opt.name == 'adam':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            if self.trained == 0:
                self.m = np.zeros((self.in_features, self.out_features))
                self.v = np.zeros((self.in_features, self.out_features))

                self.m_b = np.zeros((1, self.out_features))
                self.v_b = np.zeros((1, self.out_features))
            self.m = beta1 * self.m + (1 - beta1) * self.input.transpose().dot(h4_grad)
            self.v = beta2 * self.v + (1 - beta2) * (self.input.transpose().dot(h4_grad)** 2)
            m_hat = self.m / (1 - beta1)
            v_hat = self.v / (1 - beta2)
            self.weight = self.weight - lr * m_hat  / np.sqrt(v_hat + epsilon)
            

            self.m_b = beta1 * self.m_b + (1 - beta1) * np.sum(h4_grad, axis=0)
            self.v_b = beta2 * self.v_b + (1 - beta2) * ( np.sum(h4_grad, axis=0) ** 2)
            m_b_hat = self.m_b / (1 - beta1)
            v_b_hat = self.v_b / (1 - beta2)
            self.bias = self.bias - lr * m_b_hat  / np.sqrt(v_b_hat + epsilon)
        elif opt.name == 'SGD':
            for i in range(len(self.weight)):
                self.weight-= lr*(self.input.transpose().dot(h4_grad))
                self.bias-= lr*np.sum(h4_grad, axis=0)
        self.trained = 1
        return self.d

## by yourself .Finish your own NN framework
class relu(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        output = (np.maximum(0, input))
        return output

    def backward(self, z):
        dz = np.heaviside(z,1)
        return dz



class sigmoid():
    def __init__(self):
        pass

    def forward(self, x:np.array):
        self.inputs = x
        return self.sigmoid(x)
    
    def sigmoid(self, x:np.array):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, dy:np.array):
        dx = self.sigmoid(self.inputs) * (1 - self.sigmoid(self.inputs)) * dy
        return dx

class SoftmaxWithloss(_Layer):
    def __init__(self):
        self.soft_x=None

    def forward(self, input):      

        self.soft_x = np.zeros(shape = (input.shape[0], input.shape[1]))
        for i in range(input.shape[0]):
            self.soft_x[i] = (np.exp(input[i]-np.max(input[i]))/sum(np.exp(input[i]-np.max(input[i]))))
        predict = self.soft_x


        return predict
    def loss(self,target):
        diff = self.soft_x - target
        differences_squared = diff ** 2
        your_loss =  differences_squared.mean() 
        return your_loss

    def backward(self, a3:np.array, target:np.array):
        input_grad  = (a3-target) /len(target)
        return input_grad