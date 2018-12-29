#! python3
# -*- encoding: utf-8 -*-
# filename: test_getopt.py
# @time: 2018/9/25 13:38

import sys
import getopt


def get_options(argv):
    num_of_training = 0

    try:
        opts, args = getopt.getopt(argv[1:], "hn:", ["help", "num="])
    except getopt.GetoptError:
        print(argv[0].split('/')[-1], "-n <number of training>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print("-n <number of training>")
        elif opt in ('-n', '--num'):
            num_of_training = arg

    return num_of_training


print(get_options(sys.argv))
