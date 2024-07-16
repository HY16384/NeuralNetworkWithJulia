#https://qiita.com/kcrt/items/a7f0582a91d6599d164d

using FileIO
using Images
using ProgressBars
using JLD2

dir_array = readdir(raw"/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/extract")

data_x, data_y = Vector(), Vector()

category_num = 5

print(dir_array[1:category_num])
# count = 1

for (i, image_label) in enumerate(dir_array[1:category_num+1])
    global data_x, data_y
    println(image_label)
    if image_label!=".DS_Store"
        image_array = readdir("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/extract/"*image_label)
        for image_pass in ProgressBar(image_array)
            img = FileIO.load("/Users/yamadaharuki/Desktop/Python/NeuralNetworkWithJulia/extract/"*image_label*"/"*image_pass)
            imgarr = real.(channelview(img))
            onehot = [j==i ? 1 : 0 for j in 1:category_num]
            if size(data_x)[1] == 0
                data_x = reduce(vcat, imgarr)'
                data_y = onehot'
            else
                data_x = vcat(data_x, reduce(vcat, imgarr)')
                data_y = vcat(data_y, onehot')
            end
        end
    end
    println(size(data_x), size(data_y))
    # if size(data_x)[1] >= 10000
    #     global count
    #     open("data_x"*string(count)*".txt","w") do out
    #         Base.print_array(out, data_x)
    #     end
        
    #     open("data_y"*string(count)*".txt","w") do out
    #         Base.print_array(out, data_y)
    #     end
    #     count+=1
    #     empty!(data_x)
    #     empty!(data_y)
    # end
end

# if size(data_x) != 0
#     open("data_x"*count*".txt","w") do out
#         Base.print_array(out, data_x)
#     end
    
#     open("data_y"*count*".txt","w") do out
#         Base.print_array(out, data_y)
#     end
#     data_x, data_y = [], []
# end

@save "data_hiragana_A_x.jld2" data_x
@save "data_hiragana_A_y.jld2" data_y