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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from nltk.probability import FreqDist

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese
jieba.load_userdict("user_dict.txt")

stopwords = [word.strip() for word in codecs.open('stopword.txt', encoding='utf-8').readlines()]
root = '/Users/zhangzhen/Downloads/luckin'


def plot_cluster(result, newData, numClass, clf):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    x1 = []
    y1 = []
    for ind1 in clf.cluster_centers_:
        try:
            y1.append(ind1[1])
            x1.append(ind1[0])
        except:
            pass
    plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()


def clean(text):
    text = re.sub("@([\s\S]*?):", " ", text)  # 去除@ ...：
    text = re.sub("\[([\S\s]*?)\]", " ", text)  # [...]：
    text = re.sub("@([\s\S]*?)", "", text)  # 去除@...
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", " ", text)  # 去除标点及特殊符号
    # text = re.sub("[^\u4e00-\u9fa5]", " ", text)  # 去除所有非汉字内容（英文数字）
    return [token for token in list(jieba.lcut(text)) if token != ' ' and token not in stopwords]


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
                results.append(",".join(items[:-1]))
            elif len(items) == 2:
                # print(items[0], "\n\t$", items[-1])
                results.append(items[0])
            else:
                items = line.split(',')
                if len(items) == 2:
                    # print(items[0], "\n\t$", items[-1])
                    results.append(items[0])
                elif len(items) == 1:
                    """数据有问题"""
                    if len(line) > 5:
                        # print("$", line)
                        results.append(line)
    print("P1 Total nums:", len(results))
    return results


def read_comment():
    filename = '59790-订单评论-2019-07-16+101057.csv'
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
                # print(items[1], "\n\t$", items[2])
                if len(items[2]) > 2:
                    results.append(items[2])
            elif len(items) == 1:
                # print("$", line)
                if len(line) > 2:
                    results.append(line)
            elif len(items) == 2:
                # print(items)
                if str(items).isalnum():
                    results.append(items[-1])
                else:
                    results.append(line)
            elif len(items) == 3:
                # print(items)
                if str(items[0]).isalnum():
                    # print("\t", items[-1])
                    results.append(items[-1])
                else:
                    # print("\t", items[0])
                    results.append(items[0])
            elif len(items) == 4:
                # print(items)
                if items[0] == '0' or items[0] == '1':
                    results.append("".join(items[2:]))
                else:
                    # print("\t", items[0])
                    results.append("".join(items[:-1]))
    print("P2 Total nums:", len(results))
    return results


def build_word2vec():
    # read_feedback()
    results = read_comment() + read_feedback()
    with codecs.open("word2vec.txt", encoding='utf-8', mode='w+') as f:
        for line in results:
            tokens = clean(line)
            if len(tokens) >= 3:
                f.write(" ".join(tokens))
                f.write("\n")


def freq_dist():
    results = read_comment() + read_feedback()
    words = []
    for line in results:
        words += clean(line)
    fdist = FreqDist(words)
    # fdist.plot(50, cumulative=False)
    listkey = []
    listval = []
    print(u".........统计出现最多的前40个词...............")
    print(fdist.most_common(50))
    for key, val in sorted(fdist.items(), key=lambda x: (x[1], x[0]), reverse=True)[:40]:
        listkey.append(key)
        listval.append(val)
        # print key, val, u' ',

    df = pd.DataFrame(listval, columns=[u'次数'])
    df.index = listkey
    df.plot(kind='bar')
    plt.title(u'词频统计')
    plt.show()


def build_all():
    results = read_comment() + read_feedback()
    with codecs.open("all.txt", encoding='utf-8', mode='w+') as f:
        for line in results:
            f.write(line + "\n")


if __name__ == '__main__':
    build_word2vec()
