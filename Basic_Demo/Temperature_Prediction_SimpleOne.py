import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings
import datetime
import torch.nn as nn
from sklearn import preprocessing

warnings.filterwarnings("ignore")
# Jupyter时，放开注释
# %matplotlib inline

features = pd.read_csv('temps.csv')
features.head(n=6)
# print(features.head(6))
print('数据维度：', features.shape)

# 分别得到年\月\日
years = features['year']
months = features['month']
days = features['day']

# int转str datetime格式
dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day))
         for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# print(dates)
# x = dates[:6]
# print(x)
'''
画图
默认风格、设置布局、标签值，数据“昨天、前天...“
'''
plt.style.use('fivethirtyeight')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
fig.autofmt_xdate(rotation=45)

ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')

ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')

ax3.plot(dates, features['temp_2'])
ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')

ax4.plot(dates, features['friend'])
ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)
# plt.show()
# 将不同的字符串，转换成one-hot编码
features = pd.get_dummies(features)
features.head(6)

labels = np.array(features['actual'])
features = features.drop('actual', axis=1)
features_list = list(features.columns)
features = np.array(features)
# print(features)
print(features.shape)

# 标准化、去均值
input_features = preprocessing.StandardScaler().fit_transform(features)
# print(input_features[:3])
'''
构建网络模型 - 简化版本
'''
input_size = input_features.shape[1]
# print(input_size)
hidden_size = 256
output_size = 1
batch_size = 16

my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    nn.Dropout(p=0.5),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, output_size)
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = optim.Adam(my_nn.parameters(), lr=0.001)
'''
训练网络
'''
losses = []
for i in range(100000):
    batch_loss = []
    # mini-batch
    for start in range(0, len(input_features), batch_size):
        end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
        xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
        yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
        prediction = my_nn(xx)
        loss = cost(prediction, yy)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        batch_loss.append(loss.data.numpy())

    # 打印损失
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# 训练结果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()
predict = predict.reshape(-1)

'''
可视化预测结果
'''
dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day))
         for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
print(labels)
# 创建一个表格来存日期和其对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理，再创建一个来存日期和其对应的模型预测值
months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]
years = features[:, features_list.index('year')]

test_dates = [str(int(year))+'-'+str(int(month))+'-'+str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
print('test_dates:', test_dates)

predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict})
print(predictions_data)

'''
可视化结果
'''
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
plt.plot(predictions_data['date'], predictions_data['prediction'], 'r+', label='prediction')
plt.xticks(rotation='45')
plt.legend(loc='upper right', edgecolor='green')
# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)')
plt.title('Actual and Predicted Values')
plt.show()

'''
基本结构：
1.处理数据（训练集 + 验证集）
2.神经网络模型
3.训练神经网络
4.验证
5.可视化（option）
'''