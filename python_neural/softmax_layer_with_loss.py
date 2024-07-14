import numpy as np

class SoftmaxLayerWithLoss:

    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = self.softmax(input)

        return self.output
    
    def backward(self, d_output):
        return d_output
    
    def softmax(self, input):
        output = input.copy()
        for i in range(np.shape(input)[0]):
            m = np.max(input[i])
            output[i] = np.exp(input[i] - m) / np.sum(np.exp(input[i] - m))
        return output
    
    def cross_entropy_error(self, output, output_train, batch_size):
        return -np.sum(np.log(output[np.arange(batch_size), output_train] + 1e-7)) / batch_size