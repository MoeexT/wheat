# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from PIL import Image

WIDTH = 30
HEIGHT = 30
NUM_CLASS = 3
data_dir = 'data/'


def load_single_data(dir, _label):
    imgs = os.listdir(dir)
    num = len(imgs)

    data = np.empty((num, WIDTH, HEIGHT, 3), dtype="float32")
    label = np.empty((num,), dtype="uint8")
    for i in range(num):
        img = Image.open(dir + os.sep + imgs[i])
        img = img.resize((HEIGHT, HEIGHT))
        arr = np.asarray(img, dtype="float32")
        data[i, :, :, :] = arr
        label[i] = _label
    return data, label, num


def load_all_data(dir):
    # 加载数据
    x_train0, y_train0, num = load_single_data(data_dir + dir + '/blight', 0)
    x_train1, y_train1, num1 = load_single_data(data_dir + dir + '/powdery', 1)
    x_train2, y_train2, num2 = load_single_data(data_dir + dir + '/rust', 2)

    total = num + num1 + num2
    x_train_local = np.empty((total, WIDTH, HEIGHT, 3), dtype="float32")
    y_train_local = np.empty((total,), dtype="uint8")
    for i in range(num):
        x_train_local[i, :, :, :] = x_train0[i]
        y_train_local[i] = y_train0[i]

    for i in range(num1):
        x_train_local[num + i, :, :, :] = x_train1[i]
        y_train_local[num + i] = y_train1[i]

    for i in range(num2):
        x_train_local[num + num1 + i, :, :, :] = x_train2[i]
        y_train_local[num + num1 + i] = y_train2[i]

    return x_train_local, y_train_local

def save_data():
    x_train, y_train = load_all_data('train')
    x_test, y_test = load_all_data('test')
    np.save('data/wheat/wheat.train.data.npy', x_train)
    np.save('data/wheat/wheat.train.label.npy', y_train)
    np.save('data/wheat/wheat.test.data.npy', x_test)
    np.save('data/wheat/wheat.test.label.npy', y_test)

def load_data():
    train_data = np.load('data/wheat/wheat.train.data.npy')
    # print(train_data.shape) -> (1738, 30, 30, 3)
    train_label = np.load('data/wheat/wheat.train.label.npy')
    # print(train_label.shape) -> (1738,)
    test_data = np.load('data/wheat/wheat.test.data.npy')
    # print(test_data.shape) -> (163, 30, 30, 3)
    test_label = np.load('data/wheat/wheat.test.label.npy')
    # print(test_label.shape) -> (163,)
    return (train_data, train_label), (test_data, test_label)

if __name__=='__main__':
    save_data()
