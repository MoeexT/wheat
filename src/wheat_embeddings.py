#! py -3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import tensorflow as tf
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

from util.wheat_data import load_data

WIDTH = 30
HEIGHT = 30
batch_size = 64
NUM_CLASS = 3
data_dir = 'data/'
log_dir = 'logs/wheat'


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
    model.add(Dense(256, activation='relu', name='features'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASS, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main(times):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
        np.savetxt(f, y_test)

    y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)

    tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                        batch_size=batch_size,
                                        embeddings_freq=1,
                                        embeddings_layer_names=['features'],
                                        embeddings_metadata='sprite.png',
                                        embeddings_data=x_test)
    '''
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.09,
                                          patience=5,
                                          verbose=0,
                                          mode='auto')'''
    cbks = [tb_cb]
    model = create_model()
    model.fit(x_train, y_train, 
        batch_size=batch_size, 
        callbacks=cbks, 
        epochs=times, 
        verbose=1,
        validation_data=(x_test, y_test))
    # model.save("models/wheat.models.h5")
    # print("Saved models: wheat.models.h5")
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    print("Restored models, loss: {:5.2f}%".format(100*loss))
    print("Restored models, accuracy: {:5.2f}%".format(100*acc))


if __name__ == '__main__':
    start = time.clock()
    main(100)
    end = time.clock()
    print("运行时间：", end - start)
