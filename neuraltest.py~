import numpy
from matplotlib import pyplot as plt
class NeuralNetwork:
        
    def __init__(self, x_input, y_output):
        self.minput = x_input
        self.moutput = y_output
        self.weight1 = 2 * numpy.random.rand(self.minput.shape[1],4)
        self.weight2 = 2 * numpy.random.rand(4,1)
        print("weight1 = " + str(self.weight1) + " weight2 = " + str(self.weight2))
        self.output_layer = numpy.zeros(self.moutput.shape)    
        self.hidden_layer = numpy.zeros(self.minput.shape)

    """The sigmoid function allows a return value between 0 and 1"""
    def sigmoid_function(self, x):
        return 1 / (1 + numpy.exp(-x))

    """Sigmoid derived."""
    def deriv_sigmoid(self, x):
        return self.sigmoid_function(x) * (1 - self.sigmoid_function(x))

    def training_network(self):
        for i in range(60000):
            
            #Need to do input1 * weight1 + input 2 * weight2. Forward Propigation
            """I chose the sigmoid activation function because I only intend to use 1's and 0's
               and the Sigmoid function has restricted outputs between 0 and 1.
               Althought it is much slower compared to other Activation functions it works fine here."""
            self.output_layer = self.sig_output(self.minput)
            cost_error = numpy.square(self.moutput - self.output_layer)

            """Just printing the error margin for visual purpose"""
            if i % 10000 == 0:
                print("Error margin = " + str(self.output_layer))

            """Backpropagation starts here. Find the derivatives of everything we did up above"""
            #derive cost_error using power rule
            dcost_error = 2 * (self.moutput - self.output_layer)
            #This is backpropagation. INPUT--weight1--HIDDEN LAYER--weight2--OUTPUT LAYER->ERROR_COSTX We are
            #at the X.
            #After we find the output layer we find the error from our desired output to our calculated output
            #We start working backwards. Find the derivative of cost_error and multiply it by the derivative
            #sigmoid of output_layer
            output_layerd = dcost_error * self.deriv_sigmoid(self.output_layer)
            #INPUT--weight1--HIDDEN LAYER--weight2X--OUTPUT LAYER->ERROR_COST
            #X is our new location since we're working backwards we need to find this error which is
            #weight2 with respect to output_layerd(output_layer delta)
            hidden_layer_error = output_layerd.dot(self.weight2.T)
            #Once we find the error, we find the delta of hidden_layer by multiplying the hidden_layer_error
            #with the derivative sigmoid of hidden_layer
            hidden_layerd = hidden_layer_error * self.deriv_sigmoid(self.hidden_layer)
            
            
            #Now we do gradient descent adjust the weights by the delta of each layer to teach the AI
            self.weight2 += numpy.dot(self.hidden_layer.T, output_layerd)
            #self.hidden_layer.T.dot(output_layerd)
            self.weight1 += numpy.dot(self.minput.T, hidden_layerd)
            #self.minput.T.dot(hidden_layerd)

    def sig_output(self, input_to_use):
        self.hidden_layer = self.sigmoid_function(numpy.dot(input_to_use, self.weight1))
        new_output_layer = self.sigmoid_function(numpy.dot(self.hidden_layer, self.weight2))
        return new_output_layer


