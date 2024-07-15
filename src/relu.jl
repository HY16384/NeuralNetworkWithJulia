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

function forward(self::ReLU, input, is_train)
    input_cpu = zeros(size(input))
    copyto!(input_cpu, input)
    self.params["mask"] = [i < 0 for i in input_cpu]
    return MtlArray([i < 0 ? Float32(0) : Float32(i) for i in input_cpu])
end

function backward(self::ReLU, d_output)
    d_output_cpu = zeros(size(d_output))
    return MtlArray([is_minus ? Float32(0) : Float32(i) for (i, is_minus) in zip(d_output_cpu, self.params["mask"])])
end

function update_param(self::ReLU, optimizer::AdamOptimizer)
end
