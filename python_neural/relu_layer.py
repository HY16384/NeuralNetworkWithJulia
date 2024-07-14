import numpy as np

class ReLULayer:

    def __init__(self):
        self.mask = None

    def forward(self, input):
        self.mask = (input <= 0)
        output = input.copy()
        output[self.mask] = 0

        return output
    
    def backward(self, d_output):
        d_input = d_output.copy()
        d_input[self.mask] = 0
        
        return d_input