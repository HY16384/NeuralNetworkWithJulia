using Statistics
using StatsBase

#=
AdamOptimizer
=#
mutable struct AdamOptimizer
    alpha
    beta1
    beta2
    m_w
    m_b
    v_w
    v_b
    t
end

function init(self::AdamOptimizer, layers_l)
    for i in 1:(size(layers_l)[1]-1)
        push!(self.m_w, zeros((layers_l[i], layers_l[i+1])))
        push!(self.v_w, zeros((layers_l[i], layers_l[i+1])))
        push!(self.m_b, zeros((1, layers_l[i+1])))
        push!(self.v_b, zeros((1, layers_l[i+1])))
    end
end

function optimize(self::AdamOptimizer, grad_weight, grad_bias, index)
    self.t+=1
    self.m_w[index] = self.beta1 .* self.m_w[index] .+ (1 - self.beta1) .* grad_weight
    self.v_w[index] = self.beta2 .* self.v_w[index] .+ (1 - self.beta2) .* (grad_weight .* grad_weight)
    m_h_w = self.m_w[index] ./ (1 - self.beta1 ^ self.t)
    v_h_w = self.v_w[index] ./ (1- self.beta2 ^ self.t)

    diff_w = self.alpha .* m_h_w ./ (sqrt.(v_h_w) .+ 1e8)

    self.m_b[index] = self.beta1 .* self.m_b[index] .+ (1 - self.beta1) .* grad_bias
    self.v_b[index] = self.beta2 .* self.v_b[index] .+ (1 - self.beta2) .* (grad_bias .* grad_bias)
    m_h_b = self.m_b[index] ./ (1 - self.beta1 ^ self.t)
    v_h_b = self.v_b[index] ./ (1 - self.beta2 ^ self.t)
    
    diff_b = self.alpha .* m_h_b ./ (sqrt.(v_h_b) .+ 1e8)

    return diff_w, diff_b
end

#=
Affine Layer
=#
mutable struct Affine
    input_size
    output_size
    params
    index
end

function init(self::Affine, is_with_relu=true)
    if is_with_relu
        self.params["weight"] = randn(self.input_size, self.output_size) ./ sqrt(self.input_size) .* sqrt(2)
    else
        self.params["weight"] = randn(self.input_size, self.output_size) ./ sqrt(self.input_size)
    end
    self.params["bias"] = zeros(1, self.output_size)
end

function forward(self::Affine, input)
    self.params["input"] = input
    weight, bias = self.params["weight"], self.params["bias"]
    return input * weight .+ bias
end

function backward(self::Affine, d_output)
    weight, input = self.params["weight"], self.params["input"]
    d_input = d_output * weight'
    self.params["d_weight"] = input' * d_output
    self.params["d_bias"] = hcat([sum(d_output[:,i]) for i in 1:size(d_output)[2]])'
    return d_input
end

function update_param(self::Affine, optimizer::AdamOptimizer)
    diff_w, diff_b = optimize(optimizer, self.params["d_weight"], self.params["d_bias"], self.index)
    self.params["weight"] -= diff_w
    self.params["bias"] -= diff_b
end

#=
ReLU Layer
=#
mutable struct ReLU
    params
end

function init(self::ReLU) end

function step_func(x)
    return x >= 0 ? x : 0
end

function forward(self::ReLU, input)
    self.params["mask"] = [i < 0 for i in input]
    return [i < 0 ? 0 : i for i in input]
end

function backward(self::ReLU, d_output)
    return [is_minus ? 0 : i for (i, is_minus) in zip(d_output, self.params["mask"])]
end

function update_param(self::ReLU, optimizer::AdamOptimizer)
end

#=
Sigmoid Layer
=#

mutable struct Sigmoid
    params
end

function init(self::Sigmoid) end

function forward(self::Sigmoid, input)
    # expのoverflow対策を施した実装
    if input >=0
        output = 1 ./ (1 + exp.(-input))
    else
        output = exp.(input) ./ (1 + exp.(input))
    end
    self.params["output"] = output
    return output
end

function backward(self::Sigmoid, d_output)
    return d_output * (1.0 .- d_output) * self.params["output"]
end

function update_param(self::Sigmoid, optimizer::AdamOptimizer)
end

#=
SoftMax Layer
=#
mutable struct SoftMax
    params
end

function init(self::SoftMax) end

function forward(self::SoftMax, input)
    output = copy(input)
    for i in (1:size(input)[1])
        m = maximum(input[i, :])
        output[i, :] = exp.(input[i, :] .- m) ./ sum(exp.(input[i, :] .- m))
    end
    return output
end

function backward(self::SoftMax, d_output)
    return d_output
end

function update_param(self::SoftMax, optimizer::AdamOptimizer)
end

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

function init_neural_network(self::NeuralNet, layers_l, is_with_relu)
    self.layers = []
    self.layers_l = layers_l
    for i in 1:(size(layers_l)[1]-2)
        push!(self.layers, Affine(layers_l[i], layers_l[i+1], Dict(), i))
        if is_with_relu
            push!(self.layers, ReLU(Dict()))
        else
            push!(self.layers, Sigmoid(Dict()))
        end
    end
    push!(self.layers, Affine(layers_l[end-1], layers_l[end], Dict(), size(layers_l)[1]-1))
    push!(self.layers, SoftMax(Dict()))

    for i in 1:size(self.layers)[1]
        init(self.layers[i])
    end
end

function predict(self::NeuralNet, input)
    output = copy(input)
    for layer in self.layers
        output = forward(layer, output)
    end
    return output
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
    optimizer = AdamOptimizer(learning_rate, 0.9, 0.999, [], [], [], [], 0)
    init(optimizer, layers_l)
    for _ in 1:n_epoch
        for _ in 1:(size(x_train)[1]//batch_size + 1)
            choice = sample(1:size(x_train)[1], batch_size, replace=false)
            
            x_train_batch = [x_train[i, j] for i in choice, j in 1:layers_l[begin]]
            y_train_batch = [y_train[i, j] for i in choice, j in 1:layers_l[end]]

            prediction_batch = predict(self, x_train_batch)

            backward(self, (prediction_batch - y_train_batch) ./ batch_size)
            
            update_param(self, optimizer)
        end

        pred_train = predict(self, x_train)
        pred_val = predict(self, x_val)
        loss_train = get_loss(pred_train, y_train)
        loss_val = get_loss(pred_val, y_val)
        println("Training loss: ", loss_train, "Validation Loss: ", loss_val)
        acc_train = get_acc(pred_train, y_train)
        acc_val = get_acc(pred_val, y_train)
        println("Training accurecy: ", acc_train, "Validation accurecy: ", acc_val)
    end
end

function get_loss(prediction, target)
    return cross_entropy_error(prediction, target)
end

function get_acc(prediction, target)
    count = 0
    for i in 1:size(prediction)[1]
        if argmax(prediction[i]) == argmax(target[i])
            count+=1
        end
    end
    return count / size(prediction)[1]
end


#TODO 7/15 0:59 学習していないので修正


network = NeuralNet([],[])
layers_l = [2, 2, 2]
init_neural_network(network, layers_l, true)

x_train, y_train = [[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],[1,1],[1,0],[0,1],[0,0],],[0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,]
x_test, y_test = [[1,1],[1,0],[0,1],[0,0]],[0,1,1,0]

x_train = reduce(hcat, x_train)'
x_test = reduce(hcat, x_test)'

y_train = [i==j ? 1 : 0 for i in y_train, j in 0:1]
y_test = [i==j ? 1 : 0 for i in y_test, j in 0:1]

n_val = round(Int64, (size(x_train)[1] * 0.2))

x_val, y_val = x_train[begin:n_val, :], y_train[begin:n_val, :]
x_train, y_train = x_train[n_val:end, :], y_train[n_val:end, :]

learn(network, x_train, y_train, x_val, y_val, learning_rate=0.001, batch_size=2, n_epoch=30)