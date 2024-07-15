using JLD2
using ProgressBars
x_train = load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_x.jld2", "data_x")

#TODO 7/15 17:27　メモリ不足で落ちている　なんとかする

println("load done")

println(typeof(x_train))

len = size(x_train)[1]

x_train = reshape(vcat(vcat(x_train...)...), (127*128, len))'

print(size(x_train_mat))

@save "data_x_float.jld2" x_train_mat


y_train = load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_y.jld2", "data_y")

println("load done")

println(typeof(y_train))

y_train = reduce(vcat, transpose.(x))

print(size(y_train_mat))

@save "data_y_float.jld2" y_train_mat