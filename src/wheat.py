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
from tensorflow.python.keras.utils import plot_model

from util.wheat_data import load_data

WIDTH = 100
HEIGHT = 100
NUM_CLASS = 3
data_dir = 'data/'


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
    (x_train, y_train), (x_test, y_test), (x_evl, y_evl) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)
    y_evl = tf.keras.utils.to_categorical(y_evl, num_classes=NUM_CLASS)

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
    model = create_model()
    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_test, y_test))
    # model.save("../models/wheat.models.h5")
    # print("Saved models: wheat.models.h5")
    loss, acc = model.evaluate(x_evl, y_evl, batch_size=32)
    print("Restored models, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    start = time.clock()
    main(1000)
    end = time.clock()
    print("运行时间：", end - start)
    # plot_model(create_model(), to_file="paper/resource/NewbNet.jpg", show_shapes=True)
