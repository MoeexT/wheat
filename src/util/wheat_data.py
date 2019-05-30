# -*- coding: utf-8 -*-

import os
import time
import numpy as np
from PIL import Image

WIDTH = 32
HEIGHT = 32
NUM_CLASS = 3
original_file = 'data/'
save_dir = 'data/wheat/32x32/'
load_dir = 'data/wheat/'

"""
        train   test    valid
blight  278     94      94
powdery 337     111     111
rust    525     174     174
"""

def load_single_data(dir, _label):
    imgs = os.listdir(dir)
    num = len(imgs)

    data = np.empty((num, WIDTH, HEIGHT, 1), dtype="float32")
    label = np.empty((num,), dtype="uint8")
    for i in range(num):
        img = Image.open(dir + os.sep + imgs[i]) #.convert('L')
        img = img.resize((HEIGHT, HEIGHT))
        arr = np.asarray(img, dtype="float32") #.reshape(32, 32, 1)
        # 这里reshape的原因是图片转为灰度图'L'之后，矩阵变成二维的，需要转为三维
        data[i, :, :, :] = arr
        label[i] = _label
    return data, label, num


def load_all_data(dir):
    # 加载数据
    x_train0, y_train0, num = load_single_data(original_file + dir + '/blight', 0)
    x_train1, y_train1, num1 = load_single_data(original_file + dir + '/powdery', 1)
    x_train2, y_train2, num2 = load_single_data(original_file + dir + '/rust', 2)

    total = num + num1 + num2
    x_train_local = np.empty((total, WIDTH, HEIGHT, 1), dtype="float32")
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
    x_valid, y_valid = load_all_data('valid')
    np.save(save_dir + 'wheat.train.data.npy', x_train)
    np.save(save_dir + 'wheat.train.label.npy', y_train)
    np.save(save_dir + 'wheat.test.data.npy', x_test)
    np.save(save_dir + 'wheat.test.label.npy', y_test)
    np.save(save_dir + 'wheat.valid.data.npy', x_valid)
    np.save(save_dir + 'wheat.valid.label.npy', y_valid)

def load_data(length=379, shape='30x30'):
    '''
    length: 选择要导入的数据集的长度：
        train = length*3 = test*3 = validation*3
    shape: 
        227x227x3
        100x100x3
        30x30x3
        32x32x1
    '''
    if length > 379:
        length = 379
    elif length <=0:
        length = 100
    train_data = np.load(load_dir + shape + '/wheat.train.data.npy')
    # print(train_data.shape) -> (1140, 100, 100, 3)
    train_label = np.load(load_dir + shape + '/wheat.train.label.npy')
    # print(train_label.shape) -> (1140,)
    test_data = np.load(load_dir + shape + '/wheat.test.data.npy')
    # print(test_data.shape) -> (379, 100, 100, 3)
    test_label = np.load(load_dir + shape + '/wheat.test.label.npy')
    # print(test_label.shape) -> (379,)
    valid_data = np.load(load_dir + shape + '/wheat.valid.data.npy')
    # print(valid_data.shape) -> (379, 100, 100, 3)
    valid_label = np.load(load_dir + shape + '/wheat.valid.label.npy')
    # print(valid_label.shape) -> (379,)
    return  (train_data[:length*3], train_label[:length*3]), (test_data[:length], test_label[:length]), (valid_data[:length], valid_label[:length])
    

if __name__=='__main__':
    save_data()
#    (_,_),(_,_),(data,_) =  load_data()
#    print(data[0].shape)
#    img1 = Image.fromarray(data[0][:,:,0])  # , mode="RGB"
#    img2 = Image.fromarray(np.kron(data[0][:,:,0], np.ones((6, 6))))
#    img1.show()
#    img2.show()
