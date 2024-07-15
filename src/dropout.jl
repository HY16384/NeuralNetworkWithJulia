#=
DropOut Layer
=#
mutable struct Dropout
    ratio::Float32
    mask
end

function init(self::Dropout)
end

function forward(self::Dropout, input, is_train=true)
    if is_train
        self.mask = MtlArray(rand(Float32, size(input)) .> self.ratio)
        return input .* self.mask
    else
        return input * (Float32(1.0) - self.ratio)
    end
end

function backward(self::Dropout, d_ouput)
    return d_ouput .* self.mask
end

function update_param(self::Dropout, optimizer::AdamOptimizer)
end