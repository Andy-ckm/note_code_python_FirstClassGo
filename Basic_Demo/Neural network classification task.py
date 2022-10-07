from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import torch

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "data_minist/mnist/minist.pkl.gz"

# 判断是否存在本地文件，否，DOWNLOAD
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    '''
    protocol>=1,文件需要是二进制才能打开;MNIST数据包为PKL文件格式.

 
    #创建一个字典变量
    data = {'a':[1,2,3],'b':('string','abc'),'c':'hello'}
    print(data)
 
    #以二进制方式来存储,rb,wb,wrb,ab(新建，二进制，写入)
    pic = open(r'.\testdata.pkl', 'wb')
 
    #将字典数据存储为一个pkl文件
    pickle.dump(data,pic)
    pic.close()
    '''
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

# 数据需要转换成tensor才能进行建模和训练
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())