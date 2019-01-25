#! python3
# -*- coding: utf-8 -*-

import os
import datetime
import requests
import traceback

from manage_resource import Pysql, get_all_urls, BaiduGetter

disease_list = ["小麦白粉病", "小麦叶枯病", "小麦锈病"]
path_list = ["powdery/", "blight/", "rust/"]


def download_baidu_images():
    """
    这个函数没有插入数据库功能，已过时
    :return:
    """
    for disease in disease_list:
        getter = BaiduGetter(disease, pages=6)
        urls = getter.get_urls()
        path = check_path(disease)
        name_number = 0
        for url in urls:
            file_name = (path + "/%d.jpg") % name_number
            img_response = requests.get(url)
            open(file_name, 'wb').write(img_response.content)
            print("wrote {} to {}.".format(url, file_name))
            name_number += 1
        print(len(urls))


def check_path(illness):
    path = "../../data/images/" + illness
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def save_image(url, path, file_number):
    file_name = path + file_number + ".jpg"
    try:
        img_response = requests.get(url)
        print("writing {} to {}.".format(url, file_name), end='', flush=True)
        open(file_name, 'wb').write(img_response.content)
        print("\rwriten {} to {}.".format(url, file_name))
    except Exception as e:
        traceback.print_exc(e)
        print("Save image failed...")


def manage_urls():
    """
    从 spider.get_all_urls() 获取到全部url
    然后：把url插入到数据库，下载图片保存到本地
    """
    pysql = Pysql("localhost", "root", "mq2020.", "wheat")
    for i in range(3):
        # getter = GoogleGetter(disease_list[0], pages=6)
        url_set = get_all_urls(disease_list[i], pages=6)
        path = check_path(path_list[i])
        file_number = 1
        for set_ in url_set:
            if pysql.insert(set_[0], str(file_number) + ".jpg", set_[1], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, table=i):
                save_image(set_[0], path, str(file_number))
                file_number += 1

    pysql.close()


if __name__ == '__main__':
    manage_urls()