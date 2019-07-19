#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-15 13:30
# @Author  : zhangzhen
# @Site    : 
# @File    : data.py
# @Software: PyCharm
import re
import os
import jieba
import codecs

root = 'E:/data/lucky'
# root = '/Users/zhangzhen/Downloads/luckin'





def clean(text):
    text = re.sub("@([\s\S]*?):", " ", text)  # 去除@ ...：
    text = re.sub("\[([\S\s]*?)\]", " ", text)  # [...]：
    text = re.sub("@([\s\S]*?)", "", text)  # 去除@...
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", text)  # 去除标点及特殊符号
    text = re.sub("[^\u4e00-\u9fa5]", " ", text)  # 去除所有非汉字内容（英文数字）
    return [token for token in list(jieba.cut(text)) if token != ' ']


def read_feedback():
    filename = '意见反馈查询.csv'

    results = []
    with codecs.open(root + os.sep + filename, encoding='utf-8') as f:
        count = 0
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            line = line.strip()
            items = line.split(',"')
            if len(items) > 2:
                # print(",".join(items[:-1]), "\n\t$", items[-1])
                results.append((",".join(items[:-1]), items[-1]))
            elif len(items) == 2:
                # print(items[0], "\n\t$", items[-1])
                results.append((items[0], items[-1]))
            else:
                items = line.split(',')
                if len(items) == 2:
                    # print(items[0], "\n\t$", items[-1])
                    results.append((items[0], items[-1]))
                elif len(items) == 1:
                    """数据有问题"""
                    # print("$", line)
                    pass
    print("Total nums:", len(results))
    return results


def read_comment():
    filename = '订单评论.csv'
    results = []
    with codecs.open(root + os.sep + filename, encoding='utf-8') as f:
        count = 0
        for line in f.readlines():
            count += 1
            if count == 1:
                continue
            line = line.strip()
            items = line.split(',')
            # print(line)
            if len(items) == 5:
                pass
            elif len(items) == 1:
                if len(line) > 2:
                    results.append(line)
            elif len(items) == 2:
                if str(items).isalnum():
                    results.append(items[-1])
                else:
                    results.append(line)
            elif len(items) == 3:
                if str(items[0]).isalnum():
                    # print("\t", items[-1])
                    results.append(items[-1])
                else:
                    # print("\t", items[0])
                    results.append(items[0])
            elif len(items) == 4:
                if str(items[0]).isalnum():
                    # print("\t", items[-1])
                    results.append("".join(items[2:]))
                else:
                    # print("\t", items[0])
                    results.append("".join(items[:-1]))


if __name__ == '__main__':
    read_feedback()
    # read_comment()
