# coding: utf-8
'''
Ubuntu版本
'''
import os
import sys
import time
import getopt
import datetime
import numpy as np

import tensorflow as tf
from PIL import Image
from six.moves import range
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Dense, Dropout, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD

from util.pymail import send_email
from util.wheat_data import load_data

VERSION = "v1.2.1"

WIDTH = 100
HEIGHT = 100
NUM_CLASS = 3
data_dir = 'data/' # 相对于/root/wheat 的地址，因为是在该目录下执行的run.sh
log_path = 'logs/log_1_5w/'
checkpoint_path = "checkpoints/cp{epoch:05d}.ckpt"
save_model_path = "models/"


def get_options(argv):
    num_of_training = 0

    try:
        opts, args = getopt.getopt(argv[1:], "hn:v", ["help", "num=", "version"])
    except getopt.GetoptError:
        print(argv[0].split('/')[-1], "-n <number of training>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(
"""
-h, --help      <ask for help>
-n, --num       <number of training>
-v, --version   <version of this script>
""")
        elif opt in ('-n', '--num'):
            num_of_training = arg
            return num_of_training
        elif opt in ('-v', '--version'):
            print("Wheat " + VERSION + " on Ubuntu.")
    


def create_model():
    model = Sequential()
    # 输入: 3 通道 100x100 像素图像 -> (100, 100, 3) 张量。
    # 使用 32 个大小为 3x3 的卷积滤波器。
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASS, activation='softmax'))
    #
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def main(times=100):
    (x_train, y_train), (x_test, y_test), (x_evl, y_evl) = load_data()

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)
    y_evl = tf.keras.utils.to_categorical(y_evl, num_classes=NUM_CLASS)

    # callbacks start
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_path,
                                        histogram_freq=1,
                                        write_graph=True,
                                        write_images=False)
    '''
    cp_cb = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            verbose=1,
                                            save_weights_only=True,
                                            period=500)
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.09,
                                          patience=5,
                                          verbose=0,
                                          mode='auto')
    '''
    cbks = []
    model = create_model()
    model.fit(x_train, y_train, batch_size=16, callbacks=cbks, epochs=times, validation_data=(x_test, y_test))
    model.save(save_model_path + "_times_" +str(times) + ".model.h5")  # + datetime.datetime.now().strftime('%Y-%m-%d').split('.')[0].split(' ', '_')
    score = model.evaluate(x_evl, y_evl, batch_size=32)
    return score


if __name__ == '__main__':
    option = get_options(sys.argv)
    try:
        num_train = int(option)
    except:
        pass
    if num_train == 0:
        num_train = 50
    start = time.clock()
    fit_score = main(times=num_train)
    run_time = time.clock() - start    
    print("运行时间：", run_time)
    send_email(str(num_train) + "次训练已完成。" +
                "验证准确率：" + str(fit_score) +
                "\n运行时间：" + str(run_time))
