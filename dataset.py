
import os
import gzip
import numpy as np

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# 修改以下路径以匹配您保存MNIST文件的位置
path = '.\\'

def load_minst_train_data():
    # 读取训练集
    x_train = load_mnist_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))

    y_train = load_mnist_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))

    # 确认数据已经被正确读取
    print("训练集图片数量:", x_train.shape)
    print("训练集标签数量:", y_train.shape)
    return x_train, y_train

def load_minst_test_data():
    # 读取训练集
    x_test = load_mnist_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))

    y_test = load_mnist_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))

    # 确认数据已经被正确读取
    print("测试集图片数量:", x_test.shape)
    print("测试集标签数量:", y_test.shape)
    return x_test, y_test
