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

from util.tiger_data import load_data

WIDTH = 90
HEIGHT = 60
NUM_CLASS = 11
root_dir = 'E:\\tiger\\'


def create_model():
    model = Sequential()
    # 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
    # 使用 32 个大小为 3x3 的卷积滤波器。
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 1)))
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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main(times):
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)  #
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)  #


    tb_cb = keras.callbacks.TensorBoard(log_dir=r'logs/tiger',
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

    # try:
    #     model = keras.models.load_model("models/tiger.models.h5")
    #     print("Load models: tiger.models.h5")
    # except OSError:
    model = create_model()
    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_test, y_test))
    model.save("models/tiger.models.h5")
    print("Saved models: tiger.models.h5")
    loss, acc = model.evaluate(x_test, y_test, batch_size=32)
    print("Restored models, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    start = time.clock()
    main(500)
    end = time.clock()
    print("运行时间：", end - start)
