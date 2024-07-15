using Metal
#=
AdamOptimizer
https://arxiv.org/pdf/1412.6980
=#
mutable struct AdamOptimizer
    alpha::Float32
    beta1::Float32
    beta2::Float32
    m_w
    m_b
    v_w
    v_b
    # m_gamma
    # v_gamma
    # m_beta
    # v_beta
    t
end

function init(self::AdamOptimizer, layers_l)
    for i in 1:(size(layers_l)[1]-1)
        push!(self.m_w, MtlArray(zeros(Float32, (layers_l[i], layers_l[i+1]))))
        push!(self.v_w, MtlArray(zeros(Float32, (layers_l[i], layers_l[i+1]))))
        push!(self.m_b, MtlArray(zeros(Float32, (1, layers_l[i+1]))))
        push!(self.v_b, MtlArray(zeros(Float32, (1, layers_l[i+1]))))
        
        # push!(self.m_gamma, 0)
        # push!(self.v_gamma, 0)
        # push!(self.m_beta, 0)
        # push!(self.v_beta, 0)
    end
end

function optimize(self::AdamOptimizer, grad_weight, grad_bias, index)
    self.t+=1
    self.m_w[index] = self.beta1 .* self.m_w[index] .+ (1 - self.beta1) .* grad_weight
    self.v_w[index] = self.beta2 .* self.v_w[index] .+ (1 - self.beta2) .* (grad_weight .* grad_weight)
    m_h_w = self.m_w[index] ./ (1 - self.beta1 ^ self.t)
    v_h_w = self.v_w[index] ./ (1- self.beta2 ^ self.t)

    diff_w = self.alpha .* m_h_w ./ (sqrt.(v_h_w) .+ Float32(1e-8))

    self.m_b[index] = self.beta1 .* self.m_b[index] .+ (1 - self.beta1) .* grad_bias
    self.v_b[index] = self.beta2 .* self.v_b[index] .+ (1 - self.beta2) .* (grad_bias .* grad_bias)
    m_h_b = self.m_b[index] ./ (1 - self.beta1 ^ self.t)
    v_h_b = self.v_b[index] ./ (1 - self.beta2 ^ self.t)
    
    diff_b = self.alpha .* m_h_b ./ (sqrt.(v_h_b) .+ Float32(1e-8))

    return diff_w, diff_b
end

# function optimize_batchnorm(self::AdamOptimizer, grad_gamma, grad_beta, index)
#     self.t+=1
#     println("size ",size(grad_gamma))
#     self.m_gamma[index] = self.beta1 * self.m_gamma[index] + (1 - self.beta1) * grad_gamma
#     self.v_gamma[index] = self.beta2 * self.v_gamma[index] + (1 - self.beta2) * (grad_gamma * grad_gamma)
#     m_h_gamma = self.m_gamma[index] / (1 - self.beta1 ^ self.t)
#     v_h_gamma = self.v_gamma[index] / (1- self.beta2 ^ self.t)

#     diff_gamma = self.alpha * m_h_gamma / (sqrt(v_h_gamma) + Float32(1e-8))

#     self.m_beta[index] = self.beta1 * self.m_beta[index] + (1 - self.beta1) * grad_beta
#     self.v_beta[index] = self.beta2 * self.v_beta[index] + (1 - self.beta2) * (grad_beta * grad_beta)
#     m_h_beta = self.m_beta[index] / (1 - self.beta1 ^ self.t)
#     v_h_beta = self.v_beta[index] / (1- self.beta2 ^ self.t)

#     diff_beta = self.alpha * m_h_beta / (sqrt(v_h_beta) + Float32(1e-8))

#     return diff_gamma, diff_beta
# end