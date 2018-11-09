import numpy
# scipy.special for the sigmoid function expit()
import scipy.special

import matplotlib.pyplot as plt

# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

def main():
    input_list = []
    target_list = []

    # Number of Epoch
    epoch = 1000

    # learning rate
    learing_rate = 2

    # Inputs & Targets
    input_list.append([-1, -1]); target_list.append(0)
    input_list.append([-1, 1]); target_list.append(1)
    input_list.append([1, -1]); target_list.append(1)
    input_list.append([1, 1]); target_list.append(0)

    # Create an instance of neuralNetwork with the learning rate specified
    nn = neuralNetwork(3, 2, 1, learing_rate)
    
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
