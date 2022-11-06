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
        # 按顺序对原始图片做增强
        transforms.Compose([
            # H * W
            # 较为合理的尺寸64, 128, 256, 224
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
            # 场景为：过度曝光,窄距，光线不足...
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1,
                                   hue=0.1),
            # 概率转换成灰度率，eg.RGB --> RRR or RRB ...
            transforms.RandomGrayscale(p=0.025),
            # 转为Tensor
            transforms.ToTensor(),
            # 均值，标准差(借鉴imageNet的参数)
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
# 计算训练集和验证集的个数
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

# 读取标签对应的实际名字

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# 加载现有的模型，并直接使用其已训练的权重当初始化参数
# 常用的特征提取器，backbone
model_name = 'resnet'
feature_extract = True

# 使用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型参数的更新与否
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

model_ft = models.resnet152()

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):

    model_ft = models.resnet152(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    # 类别数
    model_ft.fc = nn.Linear(num_ftrs, 102)

    # 根据配置
    input_size = 64
    return model_ft, input_size

# 设置神经网络中需要训练的层
models_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU训练
model_ft = model_ft.to(device)

# 保存训练模型
filename = 'checkpoint.path'

# 是否训练所有层
params_to_update = model_ft.parameters()
print("params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# 优化器
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# 训练模块
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt'):
    # 计算时间
    since = time.time()
    # 记录最好的一次
    best_acc = 0
    #在GPU中训练模型
    model.to(device)
    # 训练过程中打印损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]
    # 初始化模型
    best_model_wts = copy.deepcopy(model.state_dict())
    # 遍历
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                # 训练
                model.train()
            else:
                # 验证
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # 将每个数据都取出来
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 计算损失
                running_loss += loss.item() * inputs.size(0)
                # 判断预测结果最大的和真实值是否一致
                running_corrects += torch.sum(preds == labels.data)
            #算平均
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # 每epoch 需要的时间
            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    # 字典里key就是各层的名字，值就是训练好的权重
                    'state_dict':model.state_dict(),
                    'best_acc':best_acc,
                    'optimizer':optimizer.state_dict()
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # 学习率衰减
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        # 学习衰减率
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=20)

for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

