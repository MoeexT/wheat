# -*- coding: utf-8 -*-

"""
处理已手动挑选好的图片
选取一定大小（300px*300px）保存为训练或测试集
"""

import random
import os
from PIL import Image

read_dir = r"C:\Users\Administrator\OneDrive - business\文档\JetBrains\PycharmProjects\wheat\data\test\blight\\"
save_dir = r"C:\Users\Administrator\OneDrive - business\文档\JetBrains\PycharmProjects\wheat\data\test\cut-rust\\"
read_count = 0
save_count = 0
error_count = 0

W = 210  # 切割出来的宽高
H = 210


def cut(image):
    global error_count
    img = Image.open(image).convert('RGB')
    width, height = img.size
    imgs = []
    if width >= W and height >= H:
        for i in range(3):
            x = random.randint(0, width - W)
            y = random.randint(0, height - H)
            imgs.append(img.crop((x, y, x + W, y + H)))
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
    global read_count, error_count
    img_list = os.listdir(read_dir)
    for image in img_list:
        try:
            handled_imgs = cut(read_dir + image)
            save_(handled_imgs)
            read_count += 1
        except:
            error_count += 1
    print("handled: ", read_count)
    print("saved: ", save_count)
    print("error: ", error_count)


def reshape_all():
    for file in os.listdir(read_dir):
        img = Image.open(read_dir+file).convert('RGB').resize((300, 300)).save(read_dir+file, 'JPEG')


if __name__ == '__main__':
    reshape_all()
