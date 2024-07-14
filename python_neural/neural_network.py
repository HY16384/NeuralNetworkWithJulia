import numpy as np
from affin_layer import AffineLayer
from relu_layer import ReLULayer
from softmax_layer_with_loss import SoftmaxLayerWithLoss
import matplotlib.pyplot as plt

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.params = [None] * 2
        self.params[0] = [0.01 * np.random.randn(input_size, hidden_size), np.zeros(hidden_size)]
        self.params[1] = [0.01 * np.random.randn(hidden_size, output_size), np.zeros(output_size)]

        self.layers = [None] * 4
        self.layers[0] = AffineLayer(self.params[0][0], self.params[0][1])
        self.layers[1] = ReLULayer()
        self.layers[2] = AffineLayer(self.params[1][0], self.params[1][1])
        self.layers[3] = SoftmaxLayerWithLoss()

    def predict(self, input):
        output = input
        for layer in self.layers:
              output =  layer.forward(output)
        return output

    def get_loss(self, output, output_train, batch_size):
        loss = self.layers[-1].cross_entropy_error(output, output_train, batch_size)
        return loss

    def get_accurecy(self, output, output_train):
        ans = np.argmax(output, axis=1)
        accurecy = np.sum(ans == output_train) / output.shape[0]
        return accurecy

    def get_gradient(self, output, output_train, batch_size):
        
        output_train_list = np.zeros(output.shape)
        for i, n in enumerate(output_train):
            output_train_list[i][n] = 1
        d_out = (output - output_train_list) / batch_size
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        grad = [None] * 2
        grad[0] = [self.layers[0].d_weight, self.layers[0].d_bias]
        grad[1] = [self.layers[2].d_weight, self.layers[2].d_bias]

        return grad
    
    def learn(self, input_train, output_train, learning_rate, n_iter, batch_size, show_graph=True):
        self.train_acc_l = []
        self.train_loss_l = []

        train_size = input_train.shape[0]
        for i in range(n_iter):
            batch_mask = np.random.choice(train_size, batch_size)
            input_train_batch = input_train[batch_mask]
            output_train_batch = output_train[batch_mask]

            output_pred_batch = self.predict(input_train_batch)
            loss = self.get_loss(output_pred_batch, output_train_batch, batch_size)
            accurecy = self.get_accurecy(output_pred_batch, output_train_batch)
            grad = self.get_gradient(output_pred_batch, output_train_batch, batch_size)

            for i_param in range(2):
                self.params[i_param][0] -= grad[i_param][0] * learning_rate
                self.params[i_param][1] -= grad[i_param][1] * learning_rate

            if i % 100 == 0:
                print(i)
                self.train_loss_l.append(loss)
                self.train_acc_l.append(accurecy)

        if show_graph:
            fig = plt.figure()
            ax_1 = fig.add_subplot(1, 2, 1)
            ax_2 = fig.add_subplot(1, 2, 2)
            ax_1.plot(range(n_iter//100), self.train_loss_l, label="loss")
            ax_2.plot(range(n_iter//100), self.train_acc_l, label="accurecy")
            # plt.plot(range(n_iter//100), self.train_acc_l, label="accurecy")
            plt.show()

    def save_params(self):
        pass