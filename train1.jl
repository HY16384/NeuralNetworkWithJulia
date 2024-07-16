include("src/neuralnetwork.jl")

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

using CSV
using DataFrames

network = NeuralNet([],[])
layers_l = [784, 100, 20, 10]
batchsize=32
init_neural_network(network, layers_l, true, batchsize)

train_data = CSV.read("data/mnist_train.csv", header=0, DataFrame)
test_data = CSV.read("data/mnist_test.csv", header=0, DataFrame)
x_train, y_train = train_data[1:10000, 2:end], train_data[1:10000, 1]
x_test, y_test = test_data[!, 2:end], test_data[!, 1]

println(size(x_train), size(y_train))

x_train = Matrix(x_train)
x_test = Matrix(x_test)

x_train = Float32.(x_train ./ 255)
x_test = Float32.(x_test ./ 255)

y_train = [i==j ? 1 : 0 for i in y_train, j in 0:9]
y_test = [i==j ? 1 : 0 for i in y_test, j in 0:9]

println(size(x_train), size(y_train))

n_val = round(Int64, (size(x_train)[1] * 0.2))

x_val, y_val = x_train[begin:n_val, :], y_train[begin:n_val, :]
x_train, y_train = x_train[n_val:end, :], y_train[n_val:end, :]

layers = learn(network, x_train, y_train, x_val, y_val, learning_rate=0.001, batch_size=batchsize, n_epoch=20)

save("layers.jld2", layers)