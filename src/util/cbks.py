#! py -3
# -*- coding: utf-8 -*-

import os

import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.utils import GeneratorEnqueuer, Sequence, OrderedEnqueuer
# from tensorflow.python.keras.engine.training import GeneratorEnqueuer, Sequence, OrderedEnqueuer


# from constants import batch_size

def colormap_jet(img):
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(img), 2), cv2.COLOR_BGR2RGB)

class customModelCheckpoint(Callback):
    def __init__(self, log_dir='./logs/tmp/', feed_inputs_display=None):
          super(customModelCheckpoint, self).__init__()
          self.seen = 0
          self.feed_inputs_display = feed_inputs_display
          self.writer = tf.summary.FileWriter(log_dir)

    # this function will return the feeding data for TensorBoard visualization;
    # arguments:
    #  * feed_input_display : [(input_yourModelNeed, left_image, disparity_gt ), ..., (input_yourModelNeed, left_image, disparity_gt), ...], i.e., the list of tuples of Numpy Arrays what your model needs as input and what you want to display using TensorBoard. Note: you have to feed the input to the model with feed_dict, if you want to get and display the output of your model. 
    def custom_set_feed_input_to_display(self, feed_inputs_display):
          self.feed_inputs_display = feed_inputs_display

    # copied from the above answers;
    def make_image(self, numpy_img):
          from PIL import Image
          height, width, channel = numpy_img.shape
          image = Image.fromarray(numpy_img)
          import io
          output = io.BytesIO()
          image.save(output, format='PNG')
          image_string = output.getvalue()
          output.close()
          return tf.Summary.Image(height=height, width=width, colorspace= channel, encoded_image_string=image_string)


    # A callback has access to its associated model through the class property self.model.
    def on_batch_end(self, batch, logs = None):
          logs = logs or {} 
          self.seen += 1
          if self.seen % 200 == 0: # every 200 iterations or batches, plot the costumed images using TensorBorad;
              summary_str = []
              for i in range(len(self.feed_inputs_display)):
                  feature, disp_gt, imgl = self.feed_inputs_display[i]
                  disp_pred = np.squeeze(K.get_session().run(self.model.output, feed_dict = {self.model.input : feature}), axis = 0)
                  #disp_pred = np.squeeze(self.model.predict_on_batch(feature), axis = 0)
                  summary_str.append(tf.Summary.Value(tag= 'plot/img0/{}'.format(i), image= self.make_image( colormap_jet(imgl)))) # function colormap_jet(), defined above;
                  summary_str.append(tf.Summary.Value(tag= 'plot/disp_gt/{}'.format(i), image= self.make_image( colormap_jet(disp_gt))))
                  summary_str.append(tf.Summary.Value(tag= 'plot/disp/{}'.format(i), image= self.make_image( colormap_jet(disp_pred))))

              self.writer.add_summary(tf.Summary(value = summary_str), global_step =self.seen)

#----------------------------------------------------

def get_callbacks():
    tbCallBack = TensorBoard(log_dir='./logs',
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             write_grads=True)
                             # batch_size=batch_size

    tbi_callback = TensorBoardImage('Image test')

    return [tbCallBack, tbi_callback]


def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    # height,width,channel = tensor.shape
    height, width, channel = 30, 30, 3
    print("tensor.shape: ", tensor.shape)
    print("tensor.type(): ", type(tensor))
    print("tensor: ", tensor)
    image = Image.fromarray(tensor.astype('uint8'), mode='RGB')  # TODO: maybe float ?

    output = io.BytesIO()
    image.save(output, format='JPEG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


"""
def make_image(tensor):
    # https://stackoverflow.com/questions/43784921/how-to-display-custom-images-in-tensorboard-using-keras?rq=1
    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_strimg=image_string)
"""


class TensorBoardImage(Callback):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag

    def on_epoch_end(self, epoch, logs={}):
        # Load image
        img_input = self.validation_data[0][0]  # X_train
        img_valid = self.validation_data[1][0]  # Y_train

        print(self.validation_data[0].shape)  # (8, 128, 128, 3)
        print(self.validation_data[1].shape)  # (8, 512, 512, 3)

        image = make_image(img_input)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        image = make_image(img_valid)
        summary = tf.Summary(value=[tf.Summary.Value(tag=self.tag, image=image)])
        writer = tf.summary.FileWriter('./logs')
        writer.add_summary(summary, epoch)
        writer.close()

        return


# -----------------------------------------------------------

def make_image_tensor(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Adapted from https://github.com/lanpa/tensorboard-pytorch/
    """
    if len(tensor.shape) == 3:
        height, width, channel = tensor.shape
    else:
        height, width = tensor.shape
        channel = 1
    tensor = tensor.astype(np.uint8)
    image = Image.fromarray(tensor)
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                            width=width,
                            colorspace=channel,
                            encoded_image_string=image_string)


class TensorBoardWriter:
    def __init__(self, outdir):
        assert (os.path.isdir(outdir))
        self.outdir = outdir
        self.writer = tf.summary.FileWriter(self.outdir,
                                            flush_secs=10)

    def save_image(self, tag, image, global_step=None):
        image_tensor = make_image_tensor(image)
        self.writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, image=image_tensor)]),
                                global_step)

    def close(self):
        """
        To be called in the end
        """
        self.writer.close()


class ModelDiagonoser(Callback):
    def __init__(self,
                 data_generator,
                 batch_size,
                 num_samples,
                 output_dir,
                 normalization_mean):
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.tensorboard_writer = TensorBoardWriter(output_dir)
        self.normalization_mean = normalization_mean
        is_sequence = isinstance(self.data_generator, Sequence)
        if is_sequence:
            self.enqueuer = OrderedEnqueuer(self.data_generator,
                                            use_multiprocessing=True,
                                            shuffle=False)
        else:
            self.enqueuer = GeneratorEnqueuer(self.data_generator,
                                              use_multiprocessing=True,
                                              wait_time=0.01)
        self.enqueuer.start(workers=4, max_queue_size=4)

    def on_epoch_end(self, epoch, logs=None):
        output_generator = self.enqueuer.get()
        steps_done = 0
        total_steps = int(np.ceil(np.divide(self.num_samples, self.batch_size)))
        sample_index = 0
        while steps_done < total_steps:
            generator_output = next(output_generator)
            x, y = generator_output[:2]
            y_pred = self.model.predict(x)
            y_pred = np.argmax(y_pred, axis=-1)
            y_true = np.argmax(y, axis=-1)

            for i in range(0, len(y_pred)):
                n = steps_done * self.batch_size + i
                if n >= self.num_samples:
                    return
                img = np.squeeze(x[i, :, :, :])
                img = 255. * (img + self.normalization_mean)  # mean is the training images normalization mean
                img = img[:, :, [2, 1, 0]]  # reordering of channels

                pred = y_pred[i]
                pred = pred.reshape(img.shape[0:2])

                ground_truth = y_true[i]
                ground_truth = ground_truth.reshape(img.shape[0:2])

                self.tensorboard_writer.save_image("Epoch-{}/{}/x"
                                                   .format(self.epoch_index, sample_index), img)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y"
                                                   .format(self.epoch_index, sample_index), ground_truth)
                self.tensorboard_writer.save_image("Epoch-{}/{}/y_pred"
                                                   .format(self.epoch_index, sample_index), pred)
                sample_index += 1

            steps_done += 1

    def on_train_end(self, logs=None):
        self.enqueuer.stop()
        self.tensorboard_writer.close()
