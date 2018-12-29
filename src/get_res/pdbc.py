# -*- coding: utf-8 -*-

import pymysql
import traceback


class Pysql:
    def __init__(self, host, user_name, password, database):
        """
        连接数据库，创建游标；
        提供可选表名供插入时选择
        """
        self._db = pymysql.connect(host, user_name, password, database)
        self._cursor = self._db.cursor()
        self._tables = ['wheat_blight', 'wheat_powdery', 'wheat_rust', 'images']

    def insert(self, url, file_name, source_platform, save_date, is_deleted, table=3):
        """
        :param url: 待插入的url
        :param file_name: 保存在本地的
        :param source_platform: 图片来源: Baidu, Google
        :param save_date: 插入时间
        :param is_deleted: 在本地是否被删除
        :param table: 待插入的表 {0: "blight"(叶枯病), "powdery"(白粉病), "rust"(锈病)}, 不指定则插入在默认表(测试用)
        :return:
        """
        sql = "INSERT INTO " + self._tables[table] + "(url, file_name, source_platform, save_date, is_deleted) " \
              "values (%s, %s, %s, %s, %s)"
        try:
            print("Insert {}...".format(url))
            self._cursor.execute(sql, (url, file_name, source_platform, save_date, is_deleted))
            self._db.commit()
        except Exception as e:
            print("insert failed")
            traceback.print_exc(e)
            self._db.rollback()

    def delete(self):
        pass

    def select(self):
        pass

    def close(self):
        self._cursor.close()
        self._db.close()


if __name__ == '__main__':
    import datetime
    pysql = Pysql("localhost", "root", "mq2020.", "wheat")
    pysql.insert("1", "2", "3", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0)
    pysql.close()
