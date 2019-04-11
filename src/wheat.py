#! py -3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import tensorflow as tf
from PIL import Image
from six.moves import range
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD

from util.wheat_data import load_data

WIDTH = 30
HEIGHT = 30
NUM_CLASS = 3
data_dir = 'data/'


# def load_data(dir, _label):
#     imgs = os.listdir(dir)
#     num = len(imgs)

#     data = np.empty((num, WIDTH, HEIGHT, 3), dtype="float32")
#     label = np.empty((num,), dtype="uint8")
#     for i in range(num):
#         img = Image.open(dir + os.sep + imgs[i])
#         img = img.resize((HEIGHT, HEIGHT))
#         arr = np.asarray(img, dtype="float32")
#         data[i, :, :, :] = arr
#         label[i] = _label
#     return data, label, num


# def load_train_data(dir):
#     # 加载数据
#     x_train0, y_train0, num = load_data(data_dir + dir + '/blight', 0)
#     x_train1, y_train1, num1 = load_data(data_dir + dir + '/powdery', 1)
#     x_train2, y_train2, num2 = load_data(data_dir + dir + '/rust', 2)

#     total = num + num1 + num2
#     x_train_local = np.empty((total, WIDTH, HEIGHT, 3), dtype="float32")
#     y_train_local = np.empty((total,), dtype="uint8")
#     for i in range(num):
#         x_train_local[i, :, :, :] = x_train0[i]
#         y_train_local[i] = y_train0[i]

#     for i in range(num1):
#         x_train_local[num + i, :, :, :] = x_train1[i]
#         y_train_local[num + i] = y_train1[i]

#     for i in range(num2):
#         x_train_local[num + num1 + i, :, :, :] = x_train2[i]
#         y_train_local[num + num1 + i] = y_train2[i]

#     return x_train_local, y_train_local


def create_model():
    model = Sequential()
    # 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
    # 使用 32 个大小为 3x3 的卷积滤波器。
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASS, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main(times):
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)

    # callbacks start
    tb_cb = keras.callbacks.TensorBoard(log_dir=r'..\logs',
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=False)
    '''
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.09,
                                          patience=5,
                                          verbose=0,
                                          mode='auto')'''
    cbks = [tb_cb]
    # callbacks end
    # try:
    #     model = keras.models.load_model("../models/wheat.models.h5")
    #     print("Load models: wheat.models.h5")
    # except OSError:
    model = create_model()
    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_test, y_test))
    # model.save("../models/wheat.models.h5")
    # print("Saved models: wheat.models.h5")
    loss, acc = model.evaluate(x_test, y_test, batch_size=32)
    print("Restored models, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    start = time.clock()
    main(50)
    end = time.clock()
    print("运行时间：", end - start)
