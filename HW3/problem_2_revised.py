import numpy as np

# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot as plt


class neuralNetwork:
    
    def __init__(self, learningrate):
        
        # w_i_j, from node i to node j in the next layer
        # The input to the first input node is -1 (to generate thresholds(theta))

        # [W11,W21,W31],
        # [W12,W22,W32],
        # [W13,W23,W33]
        self.wih = np.array([[1.0, 0, 0], [0.8, 0.2, 0.2], [-0.1, -0.4, -0.2]]) 

        # [W1, W2, W3]
        self.who = np.array([[0.3, 0.1, -0.4]])

        # learning rate
        self.lr = learningrate
        
        # activation function: sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
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
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_inputs))

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
    input_list = []
    target_list = []

    # Number of Epoch
    epoch = 1000

    # learning rate
    learing_rate = 1

    # Inputs & Targets
    input_list.append([-1, -1]); target_list.append(0)
    input_list.append([-1, 1]); target_list.append(1)
    input_list.append([1, -1]); target_list.append(1)
    input_list.append([1, 1]); target_list.append(0)

    # Create an instance of neuralNetwork with the learning rate specified
    nn = neuralNetwork(learing_rate)
    
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
