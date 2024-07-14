from neural_network_revised import NeuralNetworkRevised
from sklearn.datasets import fetch_openml
import numpy as np

network = NeuralNetworkRevised([784, 200, 50, 10])
mnist = fetch_openml(name='mnist_784')
input_train, output_train = mnist["data"], mnist["target"]
network.learn(
    input_train.to_numpy(),
    np.array(list(map(int, output_train))),
    0.01,
    70000,
    100,
    True)