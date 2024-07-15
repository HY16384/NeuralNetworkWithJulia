include("neuralnetwork.jl")
using JLD2

# network = NeuralNet([],[])
# layers_l = [2, 10, 4, 2]
# init_neural_network(network, layers_l, true)

# x_train, y_train = [[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],],[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,]
# x_test, y_test = [[1,1],[1,0],[0,1],[0,0]],[0,1,1,0]

# x_train = reduce(hcat, x_train)'
# x_test = reduce(hcat, x_test)'

# y_train = [i==j ? 1 : 0 for i in y_train, j in 0:1]
# y_test = [i==j ? 1 : 0 for i in y_test, j in 0:1]

# n_val = round(Int64, (size(x_train)[1] * 0.2))

# x_val, y_val = x_train[begin:n_val, :], y_train[begin:n_val, :]
# x_train, y_train = x_train[n_val:end, :], y_train[n_val:end, :]

# learn(network, x_train, y_train, x_val, y_val, learning_rate=0.001, batch_size=2, n_epoch=100)

# println(predict(network, reduce(hcat, [[1,1],[1,0],[0,1],[0,0]])'))

network = NeuralNet([],[])
layers_l = [16256, 10000, 6000, 4000, 3036]
init_neural_network(network, layers_l, true)

x_train, y_train = load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_x.jld2"), load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_y.jld2")

print(size(x_train), size(y_train))

n_val = round(Int64, (size(x_train)[1] * 0.2))

x_val, y_val = x_train[begin:n_val, :], y_train[begin:n_val, :]
x_train, y_train = x_train[n_val:end, :], y_train[n_val:end, :]

learn(network, x_train, y_train, x_val, y_val, learning_rate=0.001, batch_size=2, n_epoch=100)