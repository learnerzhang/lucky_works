#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-07-15 11:57
# @Author  : zhangzhen
# @Site    : 
# @File    : feedback.py
# @Software: PyCharm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from data import read_feedback, clean
import gensim

import jieba

import re

corpus = read_feedback()

corpus = [" ".join(clean(feed_content)) for (feed_content, opt) in corpus]
print("Case:", corpus[0])

corpus = corpus[:10000]
########################################
# TF_IDF

# 词频矩阵：矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer(max_df=0.5, stop_words=[' '])
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
freq_word_matrix = vectorizer.fit_transform(corpus)
# 获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
tfidf = transformer.fit_transform(freq_word_matrix)
# 元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
print('Features length: ' + str(len(word)))

########################################
# KMeans
from sklearn.decomposition import PCA

numClass = 7  # 聚类分几簇
clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
pca = PCA(n_components=10)  # 降维
TnewData = pca.fit_transform(weight)  # 载入N维
s = clf.fit(TnewData)


def plot_cluster(result, newData, numClass):
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


# PCA -> PLT
# pca = PCA(n_components=2)  # 输出两维
# newData = pca.fit_transform(weight)  # 载入N维
# result = list(clf.predict(TnewData))
# plot_cluster(result, newData, numClass)


# SNE
from sklearn.manifold import TSNE

# ts = TSNE(2)
# newData = ts.fit_transform(weight)
# result = list(clf.predict(TnewData))
# plot_cluster(result, newData, numClass)


newData = PCA(n_components=numClass).fit_transform(weight)  # 载入N维
newData = TSNE(2).fit_transform(newData)
result = list(clf.predict(TnewData))
plot_cluster(result, newData, numClass)
