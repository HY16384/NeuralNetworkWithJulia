using FileIO
using Images
using ProgressBars

dir_array = readdir(raw"/Users/yamadaharuki/Desktop/Python/neural/extract")

data_x, data_y = [], []

category_num = 3036

for (i, image_label) in enumerate(dir_array)
    println(image_label)
    if image_label!=".DS_Store"
        image_array = readdir("/Users/yamadaharuki/Desktop/Python/neural/extract/"*image_label)
        for image_pass in ProgressBar(image_array)
            img = FileIO.load("/Users/yamadaharuki/Desktop/Python/neural/extract/"*image_label*"/"*image_pass)
            imgarr = real.(channelview(img))
            onehot = [j==i ? 1 : 0 for j in 1:category_num]
            push!(data_x, imgarr)
            push!(data_y, onehot)
        end
    end
    if size(data_x) >= 10000
        open("data_x"*i*".txt","w") do out
            Base.print_array(out, data_x)
        end
        
        open("data_y"*i*".txt","w") do out
            Base.print_array(out, data_y)
        end
        data_x, data_y = [], []
    end
end

if size(data_x) != 0
    open("data_x"*i*".txt","w") do out
        Base.print_array(out, data_x)
    end
    
    open("data_y"*i*".txt","w") do out
        Base.print_array(out, data_y)
    end
    data_x, data_y = [], []
end