#! py -3
# -*- coding: utf-8 -*-

import io
import time

import tensorflow as tf
from PIL import Image
from tensorflow.python import keras

from util.cbks import TensorBoardImage, customModelCheckpoint
from util.models import Models
from util.wheat_data import load_data

WIDTH = 30
HEIGHT = 30
NUM_CLASS = 3
data_dir = 'data/'
checkpoint_path = 'checkpoints/'

Ishape = (32, 32, 1)
lr = 0.0001
times = 2000
model_name = 'le-net'

shape2str = str(Ishape[0]) + 'x' + str(Ishape[1])
lr2str = str(lr)
file_info = '@model '+model_name+'@epochs '+str(times)+'@shape '+str(Ishape[0])+'x'+str(Ishape[1])+' @lr '+str(lr)


def main():
    (x_train, y_train), (x_test, y_test), (x_valid, y_valid) = load_data(shape=shape2str)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASS)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASS)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes=NUM_CLASS)

    tb_cb = keras.callbacks.TensorBoard(log_dir='logs/'+file_info,
                                        write_images=False,
                                        histogram_freq=1,
                                        write_grads=False)
    # embeddings_freq=1,
    # write_images=True,
    # embeddings_data=x_train
    #
    # write_grads=True,embeddings_layer_names=None
    cp_cb = keras.callbacks.ModelCheckpoint(checkpoint_path+file_info+"/cp.ckpt",
                                            save_weights_only=True,
                                            verbose=1,
                                            period=50)
    
    es_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                          min_delta=0.09,
                                          patience=5,
                                          verbose=0,
                                          mode='auto')
    ti_cb = TensorBoardImage('Image Test')
    cbks = [tb_cb]  # , cp_cb

    model = Models.le_net(Ishape)
    model.fit(x_train, y_train, batch_size=32, callbacks=cbks, epochs=times, validation_data=(x_valid, y_valid))
    # model.save("models/wheat"+file_info+".h5")
    loss, acc = model.evaluate(x_test, y_test, batch_size=32)
    print("Restored models, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    keras.backend.clear_session()
    start = time.clock()
    main()
    end = time.clock()
    print("运行时间：", end - start)
    # plot_model(create_model(), to_file="paper/resource/NewbNet.jpg", show_shapes=True)
