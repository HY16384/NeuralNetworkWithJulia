#=
Sigmoid Layer
=#

mutable struct Sigmoid
    params
end

function init(self::Sigmoid) end

function forward(self::Sigmoid, input, is_train)
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
