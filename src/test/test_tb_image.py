import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt
import cv2
import numpy as np



def main():
    K.clear_session()
    try:
        model = load_model('models/wheat.models.h5')
    except:
        model = tf.keras.models.load_model('models/wheat.models.h5')

    images=cv2.imread("data/test.jpg")
    # cv2.imshow("Image", images)
    cv2.waitKey(0)

    # Turn the image into an array.
    # 根据载入的训练好的模型的配置，将图像统一尺寸
    image_arr = cv2.resize(images, (30, 30))

    image_arr = np.expand_dims(image_arr, axis=0)

    # 第一个 model.layers[0],不修改,表示输入数据；
    # 第二个model.layers[ ],修改为需要输出的层数的编号[]
    layer_1 = K.function([model.layers[0].input], [model.layers[1].output])

    # 只修改inpu_image
    f1 = layer_1([image_arr])[0]

    # 第一层卷积后的特征图展示，输出是（1,66,66,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    for _ in range(16):
                show_img = f1[:, :, :, _]
                show_img.shape = [98, 98]
                plt.subplot(4, 4, _ + 1)
                # plt.imshow(show_img, cmap='black')
                plt.imshow(show_img, cmap='gray')
                plt.axis('off')
    plt.show()
    return

if __name__ == '__main__':
    main()