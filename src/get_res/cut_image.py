# -*- coding: utf-8 -*-

import random
import os
from PIL import Image

read_dir = r"D:\DeepLearningProjects\images\小麦病种\叶枯病\\"
save_dir = r"D:\DeepLearningProjects\images\handled\叶枯病\\"
read_count = 0
save_count = 0
error_count = 0


def cut(image):
    global error_count
    img = Image.open(image).convert('RGB')
    width, height = img.size
    imgs = []
    if width >= 300 and height >= 300:
        for i in range(3):
            x = random.randint(0, width - 300)
            y = random.randint(0, height - 300)
            imgs.append(img.crop((x, y, x + 300, y + 300)))
    else:
        print("SizeException: ", (width, height), "From: ", image)
        error_count += 1
    return imgs


def save_(img_list):
    global save_count
    for img in img_list:
        img.save(save_dir + str(save_count) + ".jpg", 'JPEG')
        save_count += 1


def main():
    global read_count
    img_list = os.listdir(read_dir)
    for image in img_list:
        handled_imgs = cut(read_dir + image)
        save_(handled_imgs)
        read_count += 1
    print("handled: ", read_count)
    print("saved: ", save_count)
    print("error: ", error_count)


if __name__ == '__main__':
    main()
