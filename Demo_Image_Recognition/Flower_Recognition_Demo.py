import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

# 数据读取与预处理操作
data_dir = './flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

# 制作数据源：data_transforms指定了所有图像预处理操作
# ImageFolder：按文件夹的名字进行图片分类，文件夹的名字为分类的名称

data_transforms = {
    'train':
        transforms.Compose([
            # H * W
            transforms.Resize([96, 96]),
            # 随机旋转，-45到45度之间随机选
            transforms.RandomRotation(45),
            # 从中心开始裁剪
            transforms.CenterCrop(64),
            # 随机水平翻转 选择一个概率
            transforms.RandomHorizontalFlip(p=0.5),
            # 随机垂直翻转
            transforms.RandomVerticalFlip(p=0.5),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,
                                   hue=0.1),
            # 概率转换成灰度率，3通道就是R=G=B
            transforms.RandomGrayscale(p=0.025),
            # 转为Tensor
            transforms.ToTensor(),
            # 均值，标准差
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid':
        transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}
batch_size = 128

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train',
                                                                                                                'valid']
               }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

