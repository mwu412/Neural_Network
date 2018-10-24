import numpy as np

# scipy.special for the sigmoid function expit()
import scipy.special

# neural network class definition
class neuralNetwork:
    
    
    # initialise the neural network
    def __init__(self, learningrate):
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.array([[0.2, -0.4], [0.2, -0.2]]) #  [W11, W21], [W12, W22]
        self.who = np.array([0.1, -0.4])

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass

    
    # train the neural network
    def train(self, inputs_list, target): 
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2)
        targets = np.array(target, ndmin=1) #  ndmin=1 changed from the standard 
        
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
        hidden_errors = np.dot(self.who, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass

    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2)
        
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
    nn = neuralNetwork(0.1)
    for x in range(0, 10):
        nn.train(np.array([[-1], [-1]]), 0)
        nn.train(np.array([[-1], [1]]), 1)
        nn.train(np.array([[1], [-1]]), 1)
        nn.train(np.array([[-1], [-1]]), 0)

    print(nn.query([[-1], [-1]]))

if __name__ == '__main__':
    main()
