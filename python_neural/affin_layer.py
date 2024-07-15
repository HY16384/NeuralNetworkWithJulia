import numpy as np

class AffineLayer:

    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.input = None
        self.d_weight = None
        self.d_bias = None

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weight) + self.bias

        return output
    
    def backward(self, d_output):
        d_input = np.dot(d_output, self.weight.T)
        self.d_weight = np.dot(self.input.T, d_output)
        self.d_bias = np.sum(d_output, axis=0)

        print(self.d_weight, self.d_bias)

        return d_input