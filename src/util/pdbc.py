#! python3
# -*- coding: utf-8 -*-

"""
对于图片链接的增删查改
"""

import uuid
import pymysql
from requests.utils import unquote


class Pysql:
    def __init__(self, host="localhost", user_name="root", password="mq2020.", database="wheat"):
        """
        连接数据库，创建游标；
        提供可选表名供插入时选择
        """
        self._db = pymysql.connect(host, user_name, password, database)
        self._cursor = self._db.cursor()
        self.diseases = ['blight', 'powdery',  'rust']
        self._tables = {
            0: 'wheat_powdery',
            1: 'wheat_blight',
            2: 'wheat_rust',
            3: 'images',
            'powdery': 'wheat_powdery',
            'blight': 'wheat_blight',
            'rust': 'wheat_rust',
            'images': 'images',
        }

    def insert(self, url, file_name, source_platform, save_date, is_deleted, table=3):
        """
        :param url: 待插入的url
        :param file_name: 保存在本地的
        :param source_platform: 图片来源: Baidu, Google
        :param save_date: 插入时间
        :param is_deleted: 在本地是否被删除
        :param table: 表名，也可以写下标
        :return: bool: 是否插入成功
        """
        sql = "INSERT INTO " + self._tables[table] + "(uuid, url, file_name, source_platform, save_date, is_deleted) " \
                                                     "values (%s, %s, %s, %s, %s, %s)"
        try:
            print("Inserting {}...".format(url), end='', flush=True)
            self._cursor.execute(sql, (str(uuid.uuid1()), url, file_name, source_platform, save_date, is_deleted))
            self._db.commit()
            print("\rInserted {}...{}.".format(url[:6], url[-6:]))
            return True
        except Exception as e:
            print("Insert record failed: ", str(e.args))
            self._db.rollback()
            return False

    def delete(self, table, delete=False, **kwargs):
        """
        :param table: 表名
        :param delete: True: 删掉记录, False: 标记已删除
        :return:
        """
        condition_key = list(kwargs.keys())[0]
        condition_value = list(kwargs.values())[0]

        if delete:
            sql = "DELETE FROM " + self._tables[table] + " WHERE " + condition_key + "='" + condition_value + "';"
        else:
            sql = "UPDATE " + self._tables[
                table] + " SET is_deleted=1 WHERE " + condition_key + "='" + condition_value + "';"
        try:
            print("Deleting record...", end='', flush=True)
            self._cursor.execute(sql)
            self._db.commit()
            print("\rDelete record success.")
        except Exception as e:
            print("Delete record failed: ", e.args)
            self._db.rollback()

    def select(self, table, condition, *args):
        """

        %
        "url LIKE '%/%%' ESCAPE '/'" 这个条件是为了查询含有中文编码的url
        %
        :param table: 查询的表
        :param condition: 查询条件
        :param args: 查询的列名
        :return: <generator: tuple> 元素是查询结果的[元组]
        """
        args_list = [arg for arg in args]
        sql = "SELECT " + str(args_list).replace('[', '').replace(']', '').replace("'", '') + " FROM " \
              + self._tables[table] + " WHERE " + condition + ";"

        try:
            print(sql)
            print("Querying record...", end='', flush=True)
            self._cursor.execute(sql)
            print("\rQuery record success.")
            for row in self._cursor.fetchall():
                yield row[0]
        except Exception as e:
            print("Select failed: ", str(e.args))

    def _generate_update_sql(self, table, **kwargs):
        # update的参数们
        keys = list(kwargs.keys())[:-1]
        values = list(kwargs.values())[:-1]
        column = ""

        # update的条件
        condition_key = list(kwargs.keys())[-1]
        condition_value = list(kwargs.values())[-1]

        for i in range(len(keys)):
            # 对待更新的值类型判断，str/int
            if type(values[i]) == 'str':
                column += keys[i] + "='" + values[i] + "',"
            else:
                column += keys[i] + "=" + str(values[i]) + ","

        return "UPDATE " + self._tables[table] + \
               " SET " + column[:-1] + \
               " WHERE " + condition_key + "='" + condition_value + "';"

    def update(self, table, **kwargs):
        """
        pysql.update("images", is_deleted=0, file_name="b.jpg")
        :param table: 表名
        :param kwargs: 需更新的列名，条件(限一个)放最后
        :return:
        """
        sql = self._generate_update_sql(table, **kwargs)
        try:
            print("Updating record...", end='', flush=True)
            self._cursor.execute(sql)
            self._db.commit()
            print("\rUpdate record success.")
        except Exception as e:
            print("\rUpdate record failed: ", e.args)
            self._db.rollback()

    def close(self):
        self._cursor.close()
        self._db.close()


def unquote_url():
    # 垃圾函数，不想改了
    pysql = Pysql()
    for disease in ['blight', 'powdery',  'rust']:
        generator = pysql.select(disease, 'uuid', 'url')
        for tup in generator:
            print(tup)
            pysql.update(disease, url=unquote(tup[1]), uuid=tup[0])

    pysql.close()


if __name__ == '__main__':
    # main函数为测试各方法所用
    # pysql = Pysql()
    #
    # pysql.close()
    pass
