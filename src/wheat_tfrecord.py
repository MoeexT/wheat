#! python3
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

from wheat import load_train_data

WIDTH = 30
HEIGHT = 30

BATCH_SIZE = 32
NUM_CLASSES = 3

train_data_path = "../data/train/train.wheat.tfrecords"
test_data_path = "../data/test/test.wheat.tfrecords"


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {"label": tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string)}

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image_raw'] = tf.decode_raw(
        parsed_features['image_raw'], tf.uint8)

    return parsed_features['image_raw'], parsed_features["label"]


def create_dataset(filepath):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filepath)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    # dataset = dataset.shuffle(SHUFFLE_BUFFER)

    # Set the batchsize
    dataset = dataset.batch(BATCH_SIZE)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, 30, 30, 3])

    # Create a one hot array for your labels
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label


def read_and_decode(filename, one_hot=True, n_class=None, is_train=None):
    """
    Return tensor to read from TFRecord
    :param filename:
    :param one_hot:
    :param n_class:
    :param is_train:
    :return:
    """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })
    # You can do more image distortion here for training data
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image.set_shape([WIDTH*HEIGHT*3])
    image = tf.reshape(image, [WIDTH, HEIGHT, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    label = tf.cast(features['label'], tf.int32)
    if one_hot and n_class:
        label = tf.one_hot(label, n_class)

    return image, label


def read_and_show(filename):
    _image = np.empty((1738, WIDTH, HEIGHT, 3), dtype="float32")
    _label = np.empty((1738,), dtype="uint8")
    filename_queue = tf.train.string_input_producer([filename])  # 读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 取出包含image和label的feature对象
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [WIDTH, HEIGHT, 3])
    label = tf.cast(features['label'], tf.int32)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(1738):
            example, l = sess.run([image, label])  # 在会话中取出image和label
            # (30, 30, 3) ()
            # <class 'numpy.ndarray'> <class 'numpy.int32'>
            # img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            # img.save(test_record_image_path + str(i) + '_''Label_' + str(l) + '.jpg')  # 保存图片
            # print(example, l)
            _image[i, :, :, :] = np.asarray(example, dtype='float32')
            _label[i] = l

        coord.request_stop()
        coord.join(threads)
        return _image, _label


def main(times):
    x_train, y_train = read_and_show(train_data_path)
    # print(x_train, y_train)
    x_test, y_test = read_and_show(test_data_path)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

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
                                          mode='auto')
    '''
    cbks = [tb_cb]
    # cbks.append(es_cb)

    # callbacks end

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
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.01, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, batch_size=32)
    print(score[0])


if __name__ == '__main__':
    main(10)
