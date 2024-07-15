using Metal
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
        self.params["weight"] = randn(Float32, self.input_size, self.output_size) ./ Float32(sqrt(self.input_size)) .* Float32(sqrt(2))

        self.params["weight"] = MtlArray(self.params["weight"])

    else
        self.params["weight"] = randn(Float32, self.input_size, self.output_size) ./ Float32(sqrt(self.input_size))

        self.params["weight"] = MtlArray(self.params["weight"])

    end
    self.params["bias"] = zeros(Float32, 1, self.output_size)

    self.params["bias"] = MtlArray(self.params["bias"])

end

function forward(self::Affine, input, is_train)
    self.params["input"] = input
    weight, bias = self.params["weight"], self.params["bias"]
    return input * weight .+ bias
end

function backward(self::Affine, d_output)
    self.params["d_weight"] = self.params["input"]' * d_output
    # self.params["d_bias"] = hcat([sum(d_output[:,i]) for i in 1:size(d_output)[2]])'
    self.params["d_bias"] = sum(d_output, dims=1)
    return d_output * self.params["weight"]'
end

function update_param(self::Affine, optimizer::AdamOptimizer)
    diff_w, diff_b = optimize(optimizer, self.params["d_weight"], self.params["d_bias"], self.index)
    self.params["weight"] .-= diff_w
    self.params["bias"] .-= diff_b
end
