import numpy as np
import struct
import matplotlib.pyplot as plt

# 训练集文件
train_images = './data_mnist/train-images.idx3-ubyte'
# 训练集标签文件
train_labels = './data_mnist/train-labels.idx1-ubyte'

# 测试集文件
test_images = './data_mnist/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels = './data_mnist/t10k-labels.idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    # 指定的格式fmt，从偏移位置offset开始解包;
    # 二进制数据,struct模块来完成打包和解包的工作
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
    # 解析文件头信息
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels):
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images):
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels):
    return decode_idx1_ubyte(idx_ubyte_file)


def run():
    train_images_1 = load_train_images()
    train_labels_1 = load_train_labels()
    test_images_1 = load_test_images()
    test_labels_1 = load_test_labels()

    return train_images_1, train_labels_1, test_images_1, test_labels_1
'''
    # 查看前十个数据及其标签以读取是否正确
    for i in range(10):
        print(train_labels_1[i])
        print("---------------------")
        print(test_images_1[i].shape[0:2])
        print(test_labels_1[i])
        print("-------我是分隔符--------")
        plt.imshow(train_images_1[i], cmap='gray')
        plt.show()
    print('done')
'''

if __name__ == '__main__':
    run()
