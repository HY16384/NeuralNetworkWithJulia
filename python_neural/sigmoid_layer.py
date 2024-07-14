import numpy as np

class SigmoidLayer:

    def __init__(self):
        self.output = None

    def forward(self, input):
        output = 1 / (1 + np.exp(input))
        self.output = output

        return output
    
    def backward(self, d_output):
        d_input = d_output * (1.0 - d_output) * self.output

        return d_input