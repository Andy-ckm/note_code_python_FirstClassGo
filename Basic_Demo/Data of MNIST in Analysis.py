import numpy as np
import struct
import matplotlib.pyplot as plt


# 训练集文件
train_images = 'data_mnist/mnist/MNIST/raw/train-images.idx3-ubyte'
# 训练集标签文件
train_labels = 'data_mnist/mnist/MNIST/raw/train-labels.idx1-ubyte'

# 测试集文件
test_images = 'data_mnist/mnist/MNIST/raw/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels = 'data_mnist/mnist/MNIST/raw/t10k-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    # 计算给定的格式(fmt)占用多少字节的内存
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
