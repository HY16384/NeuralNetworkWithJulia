import numpy as np
from affin_layer import AffineLayer
from relu_layer import ReLULayer
from softmax_layer_with_loss import SoftmaxLayerWithLoss
import matplotlib.pyplot as plt

class NeuralNetworkRevised:

    def __init__(self, sizes_l):
        self.n_layer = np.shape(sizes_l)[0]-1
        self.params = [None] * self.n_layer
        for i in range(self.n_layer):
            self.params[i] = [0.01 * np.random.randn(sizes_l[i], sizes_l[i+1]), np.zeros(sizes_l[i+1])]

        self.layers = [None] * self.n_layer * 2
        for i in range(self.n_layer):
            self.layers[i*2] = AffineLayer(self.params[i][0], self.params[i][1])
            self.layers[i*2+1] = SoftmaxLayerWithLoss() if i == (self.n_layer-1) else ReLULayer()
            

    def predict(self, input):
        output = input
        for layer in self.layers:
            output =  layer.forward(output)
        return output

    def get_loss(self, output, output_train, batch_size):
        loss = self.layers[-1].cross_entropy_error(output, output_train, batch_size)
        return loss

    def get_accurecy(self, input_test, output_test):
        output = self.predict(input_test)
        ans = np.argmax(output, axis=1)
        accurecy = np.sum(ans == output_test) / output.shape[0]
        return accurecy

    def get_gradient(self, output, output_train, batch_size):
        
        output_train_list = np.zeros(output.shape)
        for i, n in enumerate(output_train):
            output_train_list[i][n] = 1
        d_out = (output - output_train_list) / batch_size
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)

        grad = [None] * self.n_layer
        for i in range(self.n_layer):
            grad[i] = [self.layers[i*2].d_weight, self.layers[i*2].d_bias]

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
            
            grad = self.get_gradient(output_pred_batch, output_train_batch, batch_size)

            for j in range(self.n_layer):
                self.params[j][0] -= grad[j][0] * learning_rate
                self.params[j][1] -= grad[j][1] * learning_rate

            if i % 100 == 0:
                print(i)
                loss = self.get_loss(output_pred_batch, output_train_batch, batch_size)
                test_mask = np.random.choice(train_size, batch_size)
                input_test = input_train[test_mask]
                output_test = output_train[test_mask]
                accurecy = self.get_accurecy(input_test, output_test)
                self.train_loss_l.append(loss)
                self.train_acc_l.append(accurecy)
            # self.train_loss_l.append(loss)
            # self.train_acc_l.append(accurecy)

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