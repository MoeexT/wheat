#! py -3
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

WIDTH = 90
HEIGHT = 60
NUM_CLASS = 5
root_dir = 'E:\\tiger\\'


def load_data_from_img(category):
    category_dir = root_dir + category
    categories = os.listdir(category_dir)
    num = 0
    for d in categories:
        num += len(os.listdir(category_dir + os.sep + d))

    data = np.empty((num, 1, WIDTH, HEIGHT), dtype="float32")
    label = np.empty((num,), dtype="uint8")
    n = 0
    for d in categories:
        _label = categories.index(d)
        No_dir = category_dir + os.sep + d
        for file in os.listdir(No_dir):
            img = Image.open(No_dir + os.sep + file).resize((WIDTH, HEIGHT)).convert('L')
            # print(img.size)
            arr = np.asarray(img, dtype='float32').T
            img2 = np.zeros((WIDTH, HEIGHT))
            for i in range(WIDTH):
                for j in range(HEIGHT):
                    img2[i, j] = 1 if arr[i,j] > 90 else 0
            data[n, :, :, :] = img2
            label[n] = _label
            n += 1

    data = data.reshape((num, WIDTH, HEIGHT, 1))
    return data, label


def load_data():
    train_data = np.load('data/tiger/tiger.train.data.npy')
    train_label = np.load('data/tiger/tiger.train.label.npy')
    test_data = np.load('data/tiger/tiger.test.data.npy')
    test_label = np.load('data/tiger/tiger.test.label.npy')
    return (train_data, train_label), (test_data, test_label)


def save_data():
    test_data, test_label = load_data_from_img('test')
    train_data, train_label = load_data_from_img('train')
    np.save('data/tiger/tiger.test.data.npy', test_data)
    np.save('data/tiger/tiger.test.label.npy', test_label)
    np.save('data/tiger/tiger.train.data.npy', train_data)
    np.save('data/tiger/tiger.train.label.npy', train_label)


if __name__ == '__main__':
    save_data()
