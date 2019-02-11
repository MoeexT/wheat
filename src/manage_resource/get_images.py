#! python3
# -*- coding: utf-8 -*-

import os
import datetime
import requests

from manage_resource.util import Pysql, get_all_urls, BaiduGetter

keyword_list = ["小麦白粉病", "小麦叶枯病", "小麦锈病"]
disease_list = ["powdery/", "blight/", "rust/"]


def download_baidu_images():
    """
    这个函数没有插入数据库功能，已过时
    仅作为示例
    :return:
    """
    for disease in keyword_list:
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


def check_path(path):
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
    except Exception:
        # traceback.print_exc(e)
        print("Save image failed...")


def manage_urls():
    """
    从 spider.get_all_urls() 获取到全部url
    然后：把url插入到数据库，下载图片保存到本地
    """
    pysql = Pysql("localhost", "root", "mq2020.", "wheat")
    for i in range(3):
        # getter = GoogleGetter(disease_list[0], pages=6)
        url_set = get_all_urls(keyword_list[i], pages=6)  # 获取到所有的url
        path = check_path("../../data/images/" + disease_list[i])  # 检查本地路径，如果不存在就创建
        file_number = 1
        for set_ in url_set:
            if pysql.insert(set_[0], str(file_number) + ".jpg", set_[1], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, table=i):
                save_image(set_[0], path, str(file_number))
                file_number += 1

    pysql.close()


def recover_images():
    """
        从数据库拉取已经过滤好的图片的url
        只是有几张图片的文件名相同，导致文件系统和数据库有些出入，原因不详
        :return: list[str]
        """
    pysql = Pysql()
    counter = 0
    for i in range(3):
        path = check_path("../../data/recover/" + disease_list[i])
        file_number = 1
        for url in pysql.select(disease_list[i][:-1], "is_deleted != 1", "url"):
            save_image(url, path, str(file_number))
            # print(url)
            file_number += 1
            counter += 1

    print(counter)
    # 无断点续传功能，所以要手动从错误处开始下载（因为加了trace_back，所以程序自动break了）
    # li = []
    # path = check_path("../../data/recover/rust/")
    # for url in pysql.select("rust", "is_deleted != 1", "url"):
    #     li.append(url)
    #
    # file_number = 99
    # for url in li[98:]:
    #     save_image(url, path, str(file_number))
    #     file_number +=1
    pysql.close()


if __name__ == '__main__':
    recover_images()
