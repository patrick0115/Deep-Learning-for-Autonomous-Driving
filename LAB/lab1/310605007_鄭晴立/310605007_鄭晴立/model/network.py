from .layer import *

class Network(object):
    def __init__(self):
   
        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.fc1 = FullyConnected(28*28, 128) ## Just an example.You can alter sample code anywhere. 
        self.act1 =  ACTIVITY1()
        self.fc2 = FullyConnected(128, 128)
        self.act2 =  ACTIVITY1()
        self.fc3 = FullyConnected(128, 47)        
        self.act3  = SoftmaxWithloss()
        # self.loss

    def forward(self, input, target):
        h1 = self.fc1.forward(input)
       
        h2 = self.act1.forward(h1)
        h2 = self.fc2.forward(h2)
        h3 = self.act2.forward(h2)
        h3 = self.fc3.forward(h3)
        pred, loss= self.act3.forward(h3,target)
        ## MSE
        # loss = (target - pred) ** 2
        loss = np.sum(loss) /  target.shape[0]
        # print(loss)
        return pred, loss 

    def backward(self,y:np.array,target):
        ## by yourself .Finish your own NN framework
        y = self.act3.backward(y,target)
        y = self.fc3.backward(y)
        y = self.act2.backward(y)
        y= self.fc2.backward(y)
        y = self.act1.backward(y)
        y = self.fc1.backward(y)

        # _ = self.fc2.backward(h1_grad)

    def update(self, lr):
        ## by yourself .Finish your own NN framework
        # print("update",self.fc1.weight_grad)
        self.fc1.weight = self.fc1.weight - lr * self.fc1.weight_grad
        self.fc1.bias = self.fc1.bias - lr * self.fc1.bias_grad
        self.fc2.weight = self.fc2.weight - lr * self.fc2.weight_grad
        self.fc2.bias = self.fc2.bias - lr * self.fc2.bias_grad
        self.fc3.weight = self.fc3.weight - lr * self.fc3.weight_grad
        self.fc3.bias = self.fc3.bias - lr * self.fc3.bias_grad

        # epsilon = 1e-8
        # beta1 = 0.9 
        # beta2 = 0.999

        # if self.trained == 0:
        #     self.m = np.zeros((self.input_size, self.output_size))
        #     self.v = np.zeros((self.input_size, self.output_size))
        #     self.m_b = np.zeros((1, self.output_size))
        #     self.v_b = np.zeros((1, self.output_size))
        
        # self.m = beta1 * self.m + (1 - beta1) * dw
        # self.v = beta2 * self.v + (1 - beta2) * (dw ** 2)
        # m_hat = self.m / (1 - beta1)
        # v_hat = self.v / (1 - beta2)
        # self.weight_matrix = self.weight_matrix - lr * m_hat  / np.sqrt(v_hat + epsilon)
        

        # self.m_b = beta1 * self.m_b + (1 - beta1) * db
        # self.v_b = beta2 * self.v_b + (1 - beta2) * (db ** 2)
        # m_b_hat = self.m_b / (1 - beta1)
        # v_b_hat = self.v_b / (1 - beta2)
        # self.bias_matrix = self.bias_matrix - lr * m_b_hat  / np.sqrt(v_b_hat + epsilon)

        # raise NotImplementedError