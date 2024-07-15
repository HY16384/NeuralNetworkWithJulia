#=
SoftMax Layer
=#
mutable struct SoftMax
    params
end

function init(self::SoftMax) end

function forward(self::SoftMax, input, is_train)
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
