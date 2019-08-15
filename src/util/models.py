# -*- coding: utf-8 -*-

from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.utils import plot_model

NUM_CLASS = 3


class Models:
    def __init__(self):
        self.input_shape = (30, 30, 3)

    def newb_net(shape=(30,30,3)):
        model = Sequential()
        # 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
        # 使用 32 个大小为 3x3 的卷积滤波器。
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=shape, name="conv2d"))
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

    def le_net(shape=(28, 28, 1)):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=shape, padding='valid', activation='relu',
                         kernel_initializer='uniform'))

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def alex_net(shape=(227, 227, 3)):
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=(4, 4), input_shape=shape, padding='valid', activation='relu',
                         kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        sgd = SGD(lr=0.000000001, decay=1e-6, momentum=0.01, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model


if __name__ == "__main__":
    plot_model(Models.newb_net(), to_file="newbnet.jpg", show_shapes=True)
