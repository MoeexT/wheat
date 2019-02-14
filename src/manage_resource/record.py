#! python3
# -*- coding: utf-8 -*- 

# 借鉴于 https://blog.csdn.net/chaipp0607/article/details/72960028

import os
import tensorflow as tf
from PIL import Image

record_path = "../../data/train/train.wheat.tfrecords"
test_record_image_path = "../../testRecord/"

WIDTH = 30
HEIGHT = 30


def write_record():
    path = "../../data/train/"
    diseases = ['blight', 'powdery', 'rust']

    writer = tf.python_io.TFRecordWriter(record_path)

    for index, disease_name in enumerate(diseases):
        for image_name in os.listdir(path + disease_name + '/'):
            full_name = path + disease_name + '/' + image_name
            image = Image.open(full_name).convert('RGB')
            image = image.resize((WIDTH, HEIGHT))
            image_raw = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img = tf.reshape(img, [WIDTH, HEIGHT, 3])  # reshape为30*30的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    print(img, label)
    # return img, label


def read_and_show(filename):
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
        for i in range(2):
            example, l = sess.run([image, label])  # 在会话中取出image和label
            # (30, 30, 3) ()
            # <class 'numpy.ndarray'> <class 'numpy.int32'>
            # img = Image.fromarray(example, 'RGB')  # 这里Image是之前提到的
            # img.save(test_record_image_path + str(i) + '_''Label_' + str(l) + '.jpg')  # 保存图片
            # print(example.shape, l.shape)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    # write_record()
    # read_and_decode(record_path)
    read_and_show(record_path)
