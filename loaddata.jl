using JLD2
using ProgressBars

x_train = load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_x.jld2", "data_x")

println("load done")


len = size(x_train)[1]

println(len)

x_train = x_train[1:71]

len = size(x_train)[1]

count = 1
nums = 1:len

x_train_chunk = Vector()

for i in ProgressBar(1:71)
    global nums, x_train_chunk, count
    x_train_chunk = vcat(x_train_chunk, reduce(vcat, x_train[i]))
    # if i % 3000 == 0
    #     save("data_x_"*string(count)*".jld2", x_train_chunk')
    #     print(size(x_train_chunk'))
    #     empty!(x_train_chunk)
    #     count+=1
    # end
end

if size(x_train_chunk)[1] != 0
    @save "data_x_"*string(count)*".jld2" x_train_chunk'
    empty!(x_train_chunk)
end

y_train = load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/data/data_y.jld2", "data_y")

println("load done")

y_train = y_train[1:71]

len = size(y_train)[1]

count = 1
nums = 1:len

y_train_chunk = Vector()

for i in ProgressBar(1:len)
    global nums, y_train_chunk, count
    a = rand(nums)
    nums = setdiff(nums, [a])
    y_train_chunk = vcat(x_train_chunk, x)
    if i % 3000 == 0
        save("data_y_"*string(count)*".jld2", y_train_chunk')
        print(size(y_train_chunk'))
        empty!(y_train_chunk)
        count+=1
    end
end

if size(y_train_chunk)[1] != 0
    @save "data_y_"*string(count)*".jld2" y_train_chunk'
    empty!(y_train_chunk)
end