import numpy as np
import scipy.special  # sigmoid function expit()
import matplotlib.pyplot as plt

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # numbers of nodes of each layer
        self.i = inputnodes + 1  # first input is for threshold
        self.j = hiddennodes
        self.k = outputnodes
        
        # --- Weight Matrix ----------------------------------------------
        # Wab: from node a to node b in the next layer
        # The input to the first input node is -1 (to generate thresholds)

        # [W11,W21, ...],
        # [W12,W22, ...],
        # ...

        # Initailizing the weight with "Xavier initialization"
        # Normal distribution with deviation = sqrt(1/#of nodes of previous layer)
        self.wij = np.random.randn(self.j, self.i)*np.sqrt(1/self.i) 
        self.wjk = np.random.randn(self.k, self.j)*np.sqrt(1/self.j) 
        # ----------------------------------------------------------------

        # learning rate
        self.lr = learningrate
        
        # activation function: sigmoid 
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass


    def train(self, inputs_list, targets_list):
        xi = np.array(inputs_list, ndmin=2).T  #input # convert list to 2d array
        targets = np.array(targets_list, ndmin=2).T
        
        xj = np.dot(self.wij, xi)
        yj = self.activation_function(xj)

        xk = np.dot(self.wjk, yj)
        yk = self.activation_function(xk)

        delta_k = (targets - yk) * yk *(1-yk)  # (self.k x 1) element-wise multiplication

        self.wjk += self.lr * np.dot(delta_k, np.transpose(xj))

        delta_j = yj * (yj-1) * np.dot(np.transpose(self.wjk), delta_k)

        self.wij += self.lr * np.dot(delta_j, np.transpose(xi))
        
        print('Wij: \n', self.wij)

        print('\nWjk: \n', self.wjk)


        pass


    def query(self, inputs_list):
        xi = np.array(inputs_list, ndmin=2).T  #input # convert list to 2d array
        
        xj = np.dot(self.wij, xi)
        yj = self.activation_function(xj)

        xk = np.dot(self.wjk, yj)
        yk = self.activation_function(xk)
        
        return yk



def main():
    input_list = []
    target_list = []

    # Number of Epoch
    epoch = 1000

    # learning rate
    learing_rate = 0.1

    # Inputs & Targets
    input_list.append([-1, -1]); target_list.append(0)
    input_list.append([-1, 1]); target_list.append(1)
    input_list.append([1, -1]); target_list.append(1)
    input_list.append([1, 1]); target_list.append(0)

    # Create an instance of neuralNetwork with the learning rate specified
    nn = neuralNetwork(2, 2, 1, learing_rate)
    
    # Add the threshold input
    for i in range(len(input_list)):
        input_list[i] = [-1] + input_list[i]

    # Plot the Sum-Squared Error - Epoch
    plt.axis([0, epoch+1, 0, 1.1])
    plt.title('Sum-Squared Error - Epoch\n Learing Rate = 0.1')
    plt.xlabel('Epoch')
    plt.ylabel('Sum-Squared Error')

    # Train & Plot
    for x in range(0, epoch):
        for i in range(len(input_list)):
            nn.train(input_list[i], target_list[i])  
        
        sum_squared_errors = 0

        for i in range(len(input_list)):
            sum_squared_errors += (nn.query(input_list[i])-target_list[i])**2

        plt.scatter(x+1, sum_squared_errors)

    plt.show()

if __name__ == '__main__':
    main()
