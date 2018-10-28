import numpy as np

# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot as plt

class neuralNetwork:
    
    def __init__(self):
        
        self.w31 = 0.2
        self.w32 = -0.4
        self.w41 = 0.2
        self.w42 = -0.2
        self.w43 = -0.4
        self.w43 = -0.4

        self.theta3 = 0.8
        self.theta4 = 0.3

        self.learningRate = 0.1
        pass


    def train(self, x1, x2, target): 
        net3 = x1*self.w31+x2*self.w32-self.theta3
        y3 = scipy.special.expit(net3)

        net4 = x1*self.w41+x2*self.w42+y3*self.w43-self.theta4
        Y = scipy.special.expit(net4)

        error4 = (target - Y)* scipy.special.expit(net4)* (1 - scipy.special.expit(net4))
        error3 = error4* self.w43* scipy.special.expit(net3)* (1 - scipy.special.expit(net3))

        self.w43 += self.learningRate* error4* y3

        self.w41 += self.learningRate* error4* x1
        self.w42 += self.learningRate* error4* x2

        self.w31 += self.learningRate* error3* x1
        self.w32 += self.learningRate* error3* x2

        self.theta3 += -self.learningRate* error3
        self.theta4 += -self.learningRate* error4
        pass

    def query(self, x1, x2):
        net3 = x1*self.w31+x2*self.w32-self.theta3
        y3 = scipy.special.expit(net3)

        net4 = x1*self.w41+x2*self.w42+y3*self.w43-self.theta4
        Y = scipy.special.expit(net4)

        return Y



def main():

    epoch = 10000

    plt.axis([0, epoch+1, 0, 1.5])
    plt.title('Sum-Squared Error - Epoch\n Learing Rate = 0.1')
    plt.xlabel('Epoch')
    plt.ylabel('Sum-Squared Error')

    for x in range(0, epoch):
        
        nn = neuralNetwork()
        nn.train(-1, -1, 0)
        nn.train(-1, 1, 1)
        nn.train(1, -1, 1)
        nn.train(1, 1, 0)

        sum_squared_errors = (0 - nn.query(-1,-1))**2+(1 - nn.query(-1,1))**2+(1 - nn.query(1,-1))**2+(0 - nn.query(1,1))**2

        plt.scatter(x+1, sum_squared_errors)

    plt.show()

    


if __name__ == '__main__':
    main()
