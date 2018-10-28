import numpy as np

# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot as plt


class neuralNetwork:
    
    def __init__(self, learningrate):
        
        # w_i_j, from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        # theta 3 is the threshold with input -1
        # [W11, W12, W13=0], [W21, W22, W23=0], [theta 31, theta 32, theta 33=1]
        self.wio = np.array([1.0, -1.0, 0.0]) 

        # learning rate
        self.lr = learningrate
        
        # activation function: sigmoid function
        self.activation_function = lambda x: 1 if x>0 else 0
        
        pass


    def train(self, inputs_list, target): 
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=1).T
        targets = np.array(target, ndmin=1).T #  ndmin=1 changed from the standard 
        
        # calculate signals into hidden layer
        final_inputs = np.dot(self.wio, inputs)
        # calculate the signals emerging from hidden layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        final_errors = targets - final_outputs
        
        # update the weights for the links between the input and hidden layers
        self.wio += self.lr * final_errors * np.transpose(inputs)
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.wio, inputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs



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
    nn = neuralNetwork(learing_rate)
    
    # Add the threshold input
    for i in range(len(input_list)):
        input_list[i].append(-1) 

    # Plot the Sum-Squared Error - Epoch
    plt.axis([0, epoch+1, 0, 3.1])
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

        plt.scatter(x+1, 3)

    plt.show()

if __name__ == '__main__':
    main()
