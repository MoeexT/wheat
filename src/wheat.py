#! py -3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime as dt
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
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3), name="conv2d"))
    model.add(Conv2D(32, (3, 3), activation='relu', name="conv2d_1"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d"))
    model.add(Dropout(0.25, name="dropout"))

    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_2"))
    model.add(Conv2D(64, (3, 3), activation='relu', name="conv2d_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="max_pooling2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(256, activation='relu', name="dense"))
    model.add(Dropout(0.5, name="dropout_2"))
    model.add(Dense(NUM_CLASS, activation='softmax', name="dense_1"))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main(times):
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=NUM_CLASS)

    # callbacks start
    tb_cb = keras.callbacks.TensorBoard(log_dir=r'logs',
                                        histogram_freq=1,
                                        write_images=True,
                                        )
                                        # embeddings_freq=1,
                                        
                                        # embeddings_data=x_train
                                        # write_graph=True,
                                        # write_grads=True,embeddings_layer_names=None
    '''
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.09,
                                          patience=5,
                                          verbose=0,
                                          mode='auto')'''
    cbks = [tb_cb]
    # callbacks end 
    model = create_model()
    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_valid, y_valid))
    # model.save("../models/wheat.models_" + dt.now().strftime("%Y-%m-%d_%H:%M:%S") + ".h5")
    # print("Saved models: wheat.models.h5")
    loss, acc = model.evaluate(x_test, y_test, batch_size=32)
    print("Restored models, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    start = time.clock()
    main(10)
    end = time.clock()
    print("运行时间：", end - start)
    # plot_model(create_model(), to_file="paper/resource/NewbNet.jpg", show_shapes=True)
