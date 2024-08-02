import struct
from typing import Optional, Callable

import numpy as np
from torch.utils.data import Dataset

from dataset import load_mnist_images, load_mnist_labels


class DealDataSet(Dataset):

    def __init__(self, root, train: bool = True, transform: Optional[Callable] = None,target_transform: Optional[Callable] = None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data, self.targets = self._load_data()

    def __getitem__(self, item):
        img, target = self.data[item], int(self.targets[item])
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    def __len__(self):
        return len(self.targets)

    def _load_data(self):
        if self.train:
            image_data = load_mnist_images(self.root + "train-images-idx3-ubyte")
            label_data = load_mnist_labels(self.root + "train-labels-idx1-ubyte")
            return image_data, label_data
        else:
            image_data = load_mnist_images(self.root + "t10k-images-idx3-ubyte")
            label_data = load_mnist_labels(self.root + "t10k-labels-idx1-ubyte")
            return image_data, label_data


def load_mnist_images(filename):
    # 打开文件
    with open(filename, 'rb') as f:
        # 读取前16个字节，这些字节包含了图像的魔法数、图片数量、行数和列数
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError('Not a valid MNIST image file!')
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        # 读取图像数据
        image_data = f.read()
        image_data = np.frombuffer(image_data, dtype=np.uint8)

        # 重塑数据为(num_images, rows, cols)
        images = image_data.reshape(num_images, rows, cols)

    return images


def load_mnist_labels(filename):
    # 打开文件
    with open(filename, 'rb') as f:
        # 读取前8个字节，这些字节包含了标签的魔法数和标签数量
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError('Not a valid MNIST label file!')
        num_labels = struct.unpack('>I', f.read(4))[0]

        # 读取标签数据
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


load_mnist_images("./t10k-images-idx3-ubyte")
