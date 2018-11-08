import numpy as np

import random
import math

# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot as plt


class neuralNetwork:
    
    def __init__(self, learningrate, extra_h):

        # extra_h: number of extra hidden layer nodes
        
        # w_i_j, from node i to node j in the next layer

        # Review: If no extra hidden layer:
        # [theta 31=1, theta 32, theta 33], [W21=0, W22, W23], [W11=0, W12, W13] 
        # self.wih = np.array([[1.0, 0.8, -0.1], [0.0, 0.2, -0.4], [0.0, 0.2, -0.2]]) 
        
        wih_size = extra_h + 3  # 3 = 2 input nodes (x and y) + 1 threshold node 
        self.wih = np.zeros((wih_size, wih_size)) 
        
        # can try random values in the future 
        initial_thresholds = 1.0
        initial_weights = 1.0

        # W11, W12, W13...

        # 1, initial_thresholds, initial_thresholds...
        # 0, initial_weights, initial_weights... 
        # 0, initial_weights, initial_weights...         
        # 0, 0, 0...
        # 0
        # 0
        # .
        # .
        # .
        self.wih[0, 0] = 1.0

        for x in range(1, extra_h):
            self.wih[1, x] = initial_thresholds
            self.wih[2, x] = initial_weights
            self.wih[3, x] = initial_weights

        # initial_thresholds, initial_weights, initial_weights...
        self.who = [initial_thresholds]
        for x in range(1, wih_size):
            self.who += [initial_weights]
        self.who = np.array(self.who, ndmin=2)

        # learning rate
        self.lr = learningrate
        
        # activation function: sigmoid function
        # self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function = lambda x: 2*scipy.special.expit(x)-1

        
        pass


    def train(self, inputs_list, target): 
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(target, ndmin=1).T #  ndmin=1 changed from the standard 
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



def main():
    # target function with 2 variables(x and y) 
    target_func = lambda x, y: math.cos(0.7*x+2.0*y) + 0.8*x*y

    # x range
    x_start = -1
    x_end = 1

    # y range
    y_start = -1
    y_end = 1

    # x, y step number in the range
    step = 100

    # Number of Epoch
    epoch = 100

    # learning rate
    learing_rate = 1

    # numbers of extra hidden nodes
    extra_h = 10

    # Inputs & Targets
    input_list =[]
    target_list = []

    x = x_start
    y = y_start

    for i in range(step):
        input_list.append([x, y])
        target_list.append(target_func(x, y))
        x += (x_end - x_start)/step
        y += (y_end - y_start)/step

    # Create an instance of neuralNetwork with the learning rate & numbers of hidden nodes specified
    nn = neuralNetwork(learing_rate, extra_h)
    
    # Add the threshold input
    for i in range(len(input_list)):
        input_list[i] = [-1] + input_list[i]

    # Add extra hidden nodes inputs = 0
    for i in range(len(input_list)):
        for k in range(extra_h):
            input_list[i] += [0]

    # Plot the Sum-Squared Error - Epoch
    # plt.axis([0, epoch+1, 0, 6])
    plt.title('Sum-Squared Error - Epoch\n Learing Rate = 1')
    plt.xlabel('Epoch')
    plt.ylabel('Sum-Squared Error')

    # Train & Plot
    for m in range(epoch):
        for i in range(len(input_list)):
            nn.train(input_list[i], target_list[i])  
        
        sum_squared_errors = 0

        for i in range(len(input_list)):
            sum_squared_errors += (nn.query(input_list[i])-target_list[i])**2

        plt.scatter(m+1, sum_squared_errors)

    plt.show()

if __name__ == '__main__':
    main()
