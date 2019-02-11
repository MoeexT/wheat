#! py -3
# -*- coding: utf-8 -*- 

import os
import csv
import sys
import time
from PIL import Image, ExifTags

# csv保存路径
csv_path = r'D:\Users\Teemo Nicolas\Downloads\22\\'
# 待处理图片路径
img_path = r'D:\Users\Teemo Nicolas\Downloads\22\right\cut\\'
# 水平、垂直增量
h_x = 0
h_y = 6


def save_csv(info_set):
    """
    保存处理信息到csv文件
    :param info_set:
    :return:
    """
    f = open(csv_path + '001.csv', 'a', newline='')
    writer = csv.writer(f)
    writer.writerows(info_set)
    f.close()


def cut(path, image, x, y, length, height):
    print("Managing: ", image)

    start = time.clock()
    img = Image.open(path + image).convert('RGB')
    w, h = img.size
    img.crop((x, y, x + length, y + height)).save(path + image[:-4] + '_1' + image[-4:])
    end = time.clock()
    print("Time: ", end - start)
    # 文件名，原图尺寸，起始点，子图尺寸，处理时间
    return [image, (w, h), (start_x, start_y), (crop_length, crop_height), end - start]


def main():
    info_list = []  # [[文件名，原图尺寸，起始点，子图尺寸，处理时间], [], ...]
    file_list = os.listdir(img_path)

    # 看截出来图的方向
    cut(img_path, file_list[len(file_list)//2], start_x, start_y, crop_length, crop_height)
    gon = '0'
    while gon not in ['', 'y', 'Y', 'yes', 'n', 'N', 'not']:
        gon = input("Enter go on...('y', 'n')\n")
        if gon in ['n', 'N', 'not']:
            os.remove(img_path + file_list[len(file_list)//2][:-4] + '_1' + file_list[len(file_list)//2][-4:])
            sys.exit(0)
        else:
            break

    # 正式开始截图
    for file in file_list:
        info = cut(img_path, file, start_x, start_y, crop_length, crop_height)
        info_list.append(info)

    # 保存信息?（鸡肋）
    if_save = input("Save the results?('y', 'n')\n")
    if if_save in ['', 'y', 'Y', 'yes']:
        save_csv(info_list)
    elif if_save in ['n', 'N', 'not']:
        pass


# 起始点坐标，PhotoShop获得
start_x = 804
start_y = 738
# 切割尺寸，PhotoShop获得
crop_length = 4238
crop_height = 2832
if __name__ == '__main__':
    main()


















