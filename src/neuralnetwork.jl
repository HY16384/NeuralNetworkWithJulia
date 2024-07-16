using Statistics
using StatsBase
using Plots
using Printf
using ProgressBars

using Metal

include("adam_optimizer.jl")
include("affine.jl")
include("relu.jl")
include("sigmoid.jl")
include("softmax.jl")
include("dropout.jl")
include("batch_norm.jl")

#=
CrossEntropy
=#
function cross_entropy_error(output, target)
    return mean(sum((-target .* log.(output)), dims = 1))
end

#=
NeuralNetwork
=#
mutable struct NeuralNet
    layers
    layers_l
end

function init_neural_network(self::NeuralNet, layers_l, is_with_relu, batchsize)
    self.layers = []
    self.layers_l = layers_l
    for i in 1:(size(layers_l)[1]-2)
        push!(self.layers, Affine(layers_l[i], layers_l[i+1], Dict(), i))
        if is_with_relu
            # push!(self.layers, BatchNorm((batchsize, layers_l[i+1]), 1, 0, Dict(), i))
            # push!(self.layers, BatchNorm((batchsize, layers_l[i+1]), Dict(), i))
            push!(self.layers, ReLU(Dict()))
        else
            push!(self.layers, Sigmoid(Dict()))
        end
        # push!(self.layers, Dropout(0.2, []))
    end
    push!(self.layers, Affine(layers_l[end-1], layers_l[end], Dict(), size(layers_l)[1]-1))
    push!(self.layers, SoftMax(Dict()))

    for i in 1:size(self.layers)[1]
        init(self.layers[i])
    end
end

function predict(self::NeuralNet, input, is_train=true)
    for layer in self.layers
        input = forward(layer, input, is_train)
    end
    return input
end

function backward(self::NeuralNet, d_output)

    for layer in reverse(self.layers)
        d_output = backward(layer, d_output)
    end
end

function update_param(self::NeuralNet, optimizer::AdamOptimizer)
    for layer in self.layers
        update_param(layer, optimizer)
    end
end

#=
params: [learning_rate, batch_size, n_epoch]
=#
function learn(self::NeuralNet, x_train, y_train, x_val, y_val; learning_rate, batch_size, n_epoch)
    layers_l = self.layers_l
    # optimizer = AdamOptimizer(learning_rate, 0.9, 0.999, [], [], [], [], [], [], [], [], 0)
    optimizer = AdamOptimizer(learning_rate, 0.9, 0.999, [], [], [], [], 0)
    init(optimizer, layers_l)

    loss_train_l = []
    acc_train_l = []
    loss_val_l = []
    acc_val_l = []

    for _ in 1:n_epoch
        size_x = size(x_train)[1]
        for _ in ProgressBar(1:(div(size_x, batch_size) + 1))
            # https://qiita.com/takilog/items/cc888db69669ad90c9e5
            choice = sample(1:size_x, batch_size, replace=false)
            
            x_train_batch = MtlArray(x_train[choice, :])
            y_train_batch = MtlArray(y_train[choice, :])

            prediction_batch = predict(self, x_train_batch)

            backward(self, (prediction_batch - y_train_batch) ./ batch_size)
            
            update_param(self, optimizer)
        end

        pred_train = predict(self, MtlArray(x_train[1:1000, :]))
        pred_val = predict(self, MtlArray(x_val))

        println("-----------------------------")

        pred_train_cpu, pred_val_cpu = zeros(size(pred_train)), zeros(size(pred_val))

        copyto!(pred_train_cpu, pred_train)
        copyto!(pred_val_cpu, pred_val)

        loss_train = get_loss(pred_train_cpu, y_train[1:1000, :])
        loss_val = get_loss(pred_val_cpu, y_val)
        push!(loss_train_l, loss_train)
        push!(loss_val_l, loss_val)
        @printf("Training loss:  %.3f, Validation loss:  %.3f\n", loss_train, loss_val)
        acc_train = get_acc(pred_train_cpu, y_train[1:1000, :])
        acc_val = get_acc(pred_val_cpu, y_val)
        push!(acc_train_l, acc_train)
        push!(acc_val_l, acc_val)
        @printf("Training accurecy:  %.3f, Validation accurecy:  %.3f\n", acc_train, acc_val)
        
        # plt_loss = plot(1:n_epoch, loss_train_l, label="loss_train")
        # plt_acc = plot(1:n_epoch, acc_train_l, label="acc_train")
        # plot = plot(
        #     plot!(plt_loss, 1:n_epoch, loss_val_l , label="loss_val"),
        #     plot!(plt_acc, 1:n_epoch, acc_val_l , label="acc_val"),
        #     layout = (1, 2)
        # )
        # savefig(plot, "result.png")
    end

    return self.layers
end

function get_loss(prediction, target)
    return cross_entropy_error(prediction, target)
end

function get_acc(prediction, target)
    count = 0
    for i in 1:size(prediction)[1]
        if argmax(prediction[i,:]) == argmax(target[i,:])
            count+=1
        end
    end
    return count / size(prediction)[1]
end