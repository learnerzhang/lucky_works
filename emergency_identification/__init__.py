#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-18 16:17
# @Author  : zhangzhen
# @Site    : 
# @File    : __init__.py.py
# @Software: PyCharm
import codecs
import os

from utils.path import DATA_PATH


def read_data():
    tags = []
    sents = []
    with codecs.open(os.path.join(DATA_PATH, "emergency_train.tsv"), encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            tokens = line.split('\t')
            sents.append(tokens[1])
            tags.append(tokens[0])
    return sents, tags
