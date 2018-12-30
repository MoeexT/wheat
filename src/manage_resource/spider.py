#! python3
# -*- coding:utf-8 -*- 

import csv
import time
import random
import requests
from requests.utils import quote
from bs4 import BeautifulSoup as bs


class Getter:
    def __init__(self, keyword, pages=5):
        self._pages = pages
        self._keyword = keyword
        self._headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                                       'like Gecko) Chrome/70.0.3538.110 Safari/537.36'}


class BaiduGetter(Getter):
    def __init__(self, keyword, pages=5):
        super(BaiduGetter, self).__init__(keyword, pages)
        self._platform = "Baidu"

    def _get_content(self):
        """
        只适用于下载百度图片
        :return: 所有图片的url
        """
        url = "https://images.baidu.com/search/acjson"
        # 每次请求返回内容的数量（30个图片的url）
        data_count = 30
        # 每页不同的parameter集中在一个列表里
        params = []
        for page in range(data_count, data_count*self._pages + data_count, data_count):
            params.append({
                'tn': 'resultjson_com',
                'ipn': 'rj',
                'ct': 201326592,
                'is': '',
                'fp': 'result',
                'queryWord': self._keyword,
                'cl': 2,
                'lm': -1,
                'ie': 'utf - 8',
                'oe': 'utf - 8',
                'adpicid': '',
                'st': -1,
                'z': '',
                'ic': 0,
                'word': self._keyword,
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': 0,
                'istype': 2,
                'qc': '',
                'nc': 1,
                # 请求的第几页
                'pn': page,
                'rn': 30,
                'gsm': 'd2',
                '1545820401941': ''
            })

        # 调试信息，第几页
        count = 0
        # 返回json格式中带有图片url的数据，字典类型
        contents = []
        for param in params:
            try:

                sleep_time = random.uniform(.5, 1.2)
                print("Sleeping for ", sleep_time, " second...")
                time.sleep(sleep_time)
                print("Requesting ", count, "th page...")
                contents.append(requests.get(url=url, headers=self._headers, params=param).json().get('data'))
            except requests.exceptions.ConnectionError as e:
                print(e.response)

        return contents

    def get_urls(self):
        """
        page_list: 一页的30条json数据列表
        data_dict: 一条数据，字典
        url_sets : 所有图片的url
        """
        url_sets = []
        for page_list in self._get_content():
            for data_list in page_list:
                if data_list.get('thumbURL'):
                    url_sets.append([data_list.get('thumbURL'), self._platform])

        return url_sets


class GoogleGetter(Getter):
    def __init__(self, keyword, pages=5):
        super(GoogleGetter, self).__init__(keyword, pages)
        self._platform = "Google"

    def get_urls(self):
        """
        获取谷歌图片链接
        :return: url_sets
        """
        url_sets = []
        url = "https://www.google.com.hk/search?q={}&newwindow=1&safe=strict&source=lnms&tbm=isch&sa=X&" \
              "ved=0ahUKEwiPwtrxnsXfAhUB87wKHXYSBXQQ_AUIDigB&biw=1238&bih=618".format(quote(self._keyword))
        response = requests.get(url=url, headers=self._headers)
        soup = bs(response.text, 'lxml')
        div_list = soup.find_all('div', class_='rg_meta')  # , attrs={"ou": re.compile("^http[.*]$.jpg")}
        for tag in div_list:
            if tag.contents:
                url_sets.append([eval(tag.contents[0]).get('ou'), self._platform])

        return url_sets


def save_url_list(disease, list_):
    with open(disease + "_url.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(list_)


def get_all_urls(disease, pages=5):
    bd_getter = BaiduGetter(disease, pages)
    bd_set = bd_getter.get_urls()
    gg_getter = GoogleGetter(disease, pages)
    gg_set = gg_getter.get_urls()
    result = bd_set + gg_set

    # 保存全部url到本地csv
    save_url_list(disease, result)

    return result


if __name__ == '__main__':
    # 调用这个函数是为了保存url到本地
    get_all_urls()
