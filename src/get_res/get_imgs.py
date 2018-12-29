#! py -3

import os
import datetime
import requests

from get_res import Pysql, BaiduGetter, GoogleGetter

disease_list = ["小麦白粉病", "小麦叶枯病", "小麦锈病"]


def check_path(illness):
    path = "D:/Users/Teemo Nicolas/Documents/Progects/JetBrainsProjects/PycharmProjects/wheat/data/images/" + illness
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def download_baidu_images():
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


def download_google_images():
    pysql = Pysql("localhost", "root", "mq2020.", "wheat")
    getter = GoogleGetter(disease_list[0], pages=6)
    url_list = getter.get_urls()
    file_name = 1
    for url in url_list:
        pysql.insert(url, str(file_name)+".jpg", "google", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0)
        file_name += 1
    pysql.close()


if __name__ == '__main__':
    download_google_images()
