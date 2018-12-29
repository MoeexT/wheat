# -*- coding: utf-8 -*-

try:
    print(1)
    raise Exception("ImageSizeException")
except Exception as e:
    print(e.args)
