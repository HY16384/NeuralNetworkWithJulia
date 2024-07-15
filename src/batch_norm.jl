#=
BatchNorm Layer
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
=#

mutable struct BatchNorm
    input_size
    # gamma
    # beta
    params
    index
end

function init(self::BatchNorm) end

function forward(self::BatchNorm, input, is_train)
    mu = Float32(1 / self.input_size[1]) * sum(input, dims=1)
    self.params["xmu"] = input .- mu
    self.params["var"] = Float32(1 / self.input_size[1]) .* sum(self.params["xmu"] .^ 2, dims=1)
    self.params["xhat"] = MtlArray(self.params["xmu"] ./ sqrt.(self.params["var"] .+ Float32(1e-8)))
    # output = self.params["xhat"] .* self.gamma .+ self.beta
    output = self.params["xhat"]
    return output
end

function backward(self::BatchNorm, d_output)
    # self.params["d_gamma"] = sum(d_output .* self.params["xhat"], dims=1)
    # println(size(self.params["d_gamma"]))
    # self.params["d_beta"] = sum(d_output, dims=1)

    # ここからよく分からない
    # dxhat = self.params["d_gamma"] * self.gamma
    dxhat = d_output

    divar = sum(dxhat .* self.params["xmu"], dims=1)

    dxmu1 = dxhat ./ sqrt.(self.params["var"] .+ Float32(1e-8))

    dsqrtvar = -1 ./ (self.params["var"] .+ Float32(1e-8)) .* divar

    dvar = Float32(0.5) .* 1 ./ sqrt.(self.params["var"] .+ Float32(1e-8)) .* dsqrtvar

    dsq = Float32(1 / self.input_size[1]) .* MtlArray(ones(Float32, self.input_size)) .* dvar

    dxmu2 = 2 .* self.params["xmu"] .* dsq

    dx1 = (dxmu1 .+ dxmu2)
    dmu = -1 * sum(dxmu1 .+ dxmu2, dims=1)

    dx2 = Float32(1 / self.input_size[1]) .* MtlArray(ones(Float32, self.input_size)) .* dmu

    return dx1 + dx2
end

function update_param(self::BatchNorm, optimizer::AdamOptimizer)
    # diff_gamma, diff_beta = optimize_batchnorm(optimizer, self.params["d_gamma"], self.params["d_beta"], self.index)
    # self.gamma -= diff_gamma
    # self.beta -= diff_beta
end
